"""MTP acceptance probe (greedy): main model decodes normally (CUDA graph); after every position the MTP layer is run
eagerly on (next token, target hidden) to keep its KV, and during generation a K-deep draft chain is made and later
scored against what the main model actually produced. torchrun --nproc-per-node 4 mtp_probe.py [--n 200] [--k 3]"""
import argparse, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["MIMO_MTP"] = "1"
import torch, jinja2, torch.distributed as dist
from tokenizers import Tokenizer
import mimo_tp as TP
M = TP.M
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--out", default="runs/mtp_probe.json")
a = ap.parse_args()
rank, W = TP.init_dist(); R0 = rank == 0
tok = Tokenizer.from_file(str(M.ASSETS / "tokenizer.json"))
tmpl = jinja2.Environment().from_string(open(M.ASSETS / "chat_template.jinja").read())
enc = lambda s: tok.encode(s, add_special_tokens=False).ids
chat = lambda s, th=False: enc(tmpl.render(messages=[{"role": "user", "content": s}], add_generation_prompt=True, enable_thinking=th))
prompts = [("story", chat("Write a short story (about 250 words) about a lighthouse keeper who finds a message in a bottle.")),
           ("code", chat("Write a Python function that merges two sorted lists into one sorted list, with a docstring and tests.")),
           ("explain", chat("Explain how a transformer language model works, in about 200 words.")),
           ("think", chat("What is 17 * 23? Think step by step.", th=True)),
           ("completion", enc("The capital of France is"))]
model = TP.TPModel(rank, W)
model.capture()
mt = model.mtp
res = {}
with torch.inference_mode():
    for name, ids in prompts:
        model.reset(); mt.layer.kc.zero_(); mt.layer.vc.zero_()
        model.st["pos"].fill_(0)
        seq = list(ids); drafts = {}
        P = len(ids); i = 0
        t0 = time.time()
        while True:
            if i < P:
                model.tok_in.fill_(seq[i])
            model.replay()
            g = model.tok_out.item()
            if i + 1 >= P:
                seq.append(g)
            nxt = torch.tensor([seq[i + 1]], device=model.dev)
            d, h = model.mtp_step(nxt, model.hf, i)
            if i + 1 >= P:
                kc, vc = mt.layer.kc.clone(), mt.layer.vc.clone()
                ch = [d.item()]
                for j in range(1, a.k):
                    d, h = model.mtp_step(d.clone(), h.clone(), i + j)
                    ch.append(d.item())
                mt.layer.kc.copy_(kc); mt.layer.vc.copy_(vc)
                drafts[i] = ch  # ch[j] predicts seq[i + 2 + j]
            i += 1
            if i + 1 >= P and (len(seq) - P >= a.n or seq[-1] in (151643, 151645)):
                break
        # score: at step i the verifier would take drafts[i-1]... use chain made after position i: drafts for i+2..
        acc = [0] * (a.k + 1); nst = 0
        for i0, ch in drafts.items():
            if i0 + 2 + a.k > len(seq):
                continue
            nst += 1
            L = 0
            while L < a.k and ch[L] == seq[i0 + 2 + L]:
                L += 1
            acc[L] += 1
        dist_ = [x / max(nst, 1) for x in acc]
        # expected tokens per verify step with k' drafts = 1 + sum_{j<k'} P(L > j)
        exp = [1 + sum(sum(dist_[l] for l in range(j + 1, a.k + 1)) for j in range(kk)) for kk in range(a.k + 1)]
        res[name] = {"steps": nst, "accept_len_dist": dist_, "expected_tokens_per_step_k": exp,
                     "text": tok.decode(seq[P:], skip_special_tokens=False)[:400], "s": time.time() - t0}
        if R0:
            M.log(f"{name}: steps {nst} accept-len dist {[round(x, 3) for x in dist_]} E[tok/step] k=0..{a.k}: {[round(x, 3) for x in exp]}")
            json.dump(res, open(a.out, "w"), indent=1)
dist.barrier(); sys.stdout.flush(); os._exit(0)

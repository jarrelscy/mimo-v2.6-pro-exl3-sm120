"""Spec-decode smoke + speed: baseline graph first (reference tokens), then Spec graphs for each K.
torchrun --nproc-per-node 4 spec_smoke.py --ks 0,1,2,3 [--ctx 2000,4000] [--out runs/spec_smoke.json]"""
import argparse, json, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MIMO_MTP", "1")
import torch, jinja2
import torch.distributed as dist
from tokenizers import Tokenizer
import mimo_tp as TP
import spec_tp as SP
M = TP.M
ap = argparse.ArgumentParser()
ap.add_argument("--ks", default="0,1,2,3")
ap.add_argument("--ctx", default="2000,4000")
ap.add_argument("--tests", default="completion,chat,long,code,ctx")
ap.add_argument("--nobase", action="store_true")
ap.add_argument("--out", default="/data/Jarrel/mimo-pro-exl3-fast/runs/spec_smoke.json")
args = ap.parse_args()
rank, W = TP.init_dist(); R0 = rank == 0
tok = Tokenizer.from_file(str(M.ASSETS / "tokenizer.json"))
tmpl = jinja2.Environment().from_string(open(M.ASSETS / "chat_template.jinja").read())
enc = lambda s: tok.encode(s, add_special_tokens=False).ids
dec = lambda ids: tok.decode(ids, skip_special_tokens=False)
chat = lambda s: enc(tmpl.render(messages=[{"role": "user", "content": s}], add_generation_prompt=True, enable_thinking=False))
sel = args.tests.split(",")
tests = []
if "completion" in sel: tests.append(("completion", enc("The capital of France is"), 32))
if "chat" in sel: tests.append(("chat", chat("In one or two sentences, what is photosynthesis?"), 96))
if "long" in sel: tests.append(("long", chat("Write a short story (about 250 words) about a lighthouse keeper who finds a message in a bottle."), 300))
if "code" in sel: tests.append(("code", chat("Write a Python function that merges two sorted lists into one sorted list, with a docstring and tests."), 300))
if "ctx" in sel:
    filler_ids = enc(open("/data/Jarrel/mimo-pro-exl3-smoke/mimo_exl3.py").read())
    needle = "\n# NOTE: the secret passphrase for this file is 'violet-harbor-417'.\n"
    for n in [int(c) for c in args.ctx.split(",") if c]:
        body = (filler_ids * (n // len(filler_ids) + 2))[: n - 120]
        half = len(body) // 2
        tests.append((f"ctx{n}", chat("Here is a source file:\n\n" + dec(body[:half]) + needle + dec(body[half:]) +
                                      "\n\nWhat is the secret passphrase mentioned in a NOTE comment in the file? Answer with just the passphrase."), 48))
t0 = time.time()
model = TP.TPModel(rank, W)
if R0: M.log(f"loaded {time.time()-t0:.0f}s mem {torch.cuda.memory_allocated()/2**30:.1f} GiB")
res = {"env": {k: v for k, v in os.environ.items() if k.startswith(("MIMO_", "NCCL_"))}, "runs": {}}
base = {}
def summ(name, ids, out, t_pre, times, acc=None):
    text = dec(out)
    st = {"prompt_tokens": len(ids), "new_tokens": len(out), "prefill_s": round(t_pre, 3), "prefill_tok_s": round(len(ids) / t_pre, 1)}
    if times:
        ts = sorted(times)
        st.update(step_ms_median=round(ts[len(ts) // 2] * 1e3, 3), steps=len(times),
                  tok_per_step=round((len(out) - 1) / len(times), 3), decode_tok_s=round((len(out) - 1) / sum(times), 2))
    if name.startswith("ctx"): st["needle"] = "violet-harbor-417" in text
    st["text"] = text
    return st
with torch.inference_mode():
    if not args.nobase:
        model.capture()
        r = {}
        for name, ids, n in tests:
            out, t_pre, times = model.generate(ids, n, return_times=True)
            base[name] = out
            r[name] = summ(name, ids, out, t_pre, times)
            if R0: M.log(f"[base] {name}: {({k: v for k, v in r[name].items() if k != 'text'})}")
        res["runs"]["base"] = r
        model.graph = None; torch.cuda.synchronize(); torch.cuda.empty_cache()
    for kk in args.ks.split(","):
        K = int(kk.rstrip("b"))  # "3b" = bf16 draft lm_head/MLP (MIMO_DRAFT_FP8=0)
        os.environ["MIMO_DRAFT_FP8"] = "0" if kk.endswith("b") else "1"
        sp = SP.Spec(model, K)
        t0 = time.time(); sp.capture(K + 1)
        if sp.PF > K + 1: sp.capture(sp.PF, pf=True)
        if R0: M.log(f"[spec K={kk}] captured {time.time()-t0:.1f}s mem {torch.cuda.memory_allocated()/2**30:.1f} GiB")
        r = {}
        for name, ids, n in tests:
            out, t_pre, times = sp.generate(ids, n, return_times=True)
            st = summ(name, ids, out, t_pre, times)
            if name in base:
                b = base[name]; same = 0
                while same < min(len(b), len(out)) and b[same] == out[same]: same += 1
                st["same_prefix_as_base"] = same; st["identical"] = b == out
            if sp.last_accept:
                a = sp.last_accept
                st["accept_hist"] = [a.count(i) for i in range(K + 1)]
            r[name] = st
            if R0: M.log(f"[spec K={kk}] {name}: {({k: v for k, v in st.items() if k != 'text'})}")
        res["runs"][f"K{kk}"] = r
        if R0: json.dump(res, open(args.out, "w"), indent=1)
        sp.graphs.clear(); del sp; torch.cuda.synchronize(); torch.cuda.empty_cache()
if R0:
    for name in res["runs"].get("K1", res["runs"].get("K0", {})):
        M.log(f"--- {name}: " + " | ".join(f"{k}: {v[name].get('decode_tok_s')} tok/s" for k, v in res["runs"].items() if name in v))
dist.barrier(); sys.stdout.flush(); os._exit(0)

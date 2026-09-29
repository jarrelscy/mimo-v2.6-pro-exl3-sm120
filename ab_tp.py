"""In-process A/B of graph-time knobs (one model load, re-capture per config).
torchrun --nproc-per-node 4 ab_tp.py --configs 'pf=0,0,0;blk=32;hot=2,1024,4 | ...' --out runs/ab.json
Keys: pf=A=q42+g9/D=o24 (',' inside pf written as '+')  blk=N (prefetch blocks)  hot=BN,BK,NW (hot gate/up)  fuse=0/1 (AR+add+rmsnorm)"""
import argparse, json, sys, os, gc, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, jinja2, torch.distributed as dist
from tokenizers import Tokenizer
import mimo_tp as TP, car
M = TP.M
ap = argparse.ArgumentParser()
ap.add_argument("--configs", required=True)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--out", default="runs/ab.json")
a = ap.parse_args()
rank, W = TP.init_dist(); R0 = rank == 0
tok = Tokenizer.from_file(str(M.ASSETS / "tokenizer.json"))
tmpl = jinja2.Environment().from_string(open(M.ASSETS / "chat_template.jinja").read())
enc = lambda s: tok.encode(s, add_special_tokens=False).ids
ids = enc(tmpl.render(messages=[{"role": "user", "content": "Write a short story (about 250 words) about a lighthouse keeper who finds a message in a bottle."}],
                      add_generation_prompt=True, enable_thinking=False))
model = TP.TPModel(rank, W)
res = []
cfgs = [c.strip() for c in a.configs.split("|")]
for rep in range(a.reps):
    for c in cfgs:
        kv = dict(p.split("=", 1) for p in c.split(";") if p)
        TP.PF.clear(); TP.PF.update(TP.parse_pf(kv.get("pf", "").replace("+", ",")))
        car.PF_BLK = int(kv.get("blk", "32"))
        TP.FUSENORM = kv.get("fuse", "0") == "1"
        car.THR1 = int(kv.get("thr1", "256"))
        TP.HOT_GU[:] = [int(v) for v in kv.get("hot", "2,1024,4").split(",")]
        model.graph = None; gc.collect(); torch.cuda.empty_cache()
        model.capture()
        out, _, times = model.generate(ids, a.n, return_times=True)
        ts = sorted(times)
        r = {"cfg": c, "rep": rep, "median_ms": ts[len(ts) // 2] * 1e3, "tok_s": len(times) / sum(times), "n": len(out),
             "text": tok.decode(out, skip_special_tokens=False)}
        res.append(r)
        if R0:
            M.log(f"[{rep}] {c}: median {r['median_ms']:.3f} ms  {r['tok_s']:.1f} tok/s  n={r['n']}  same_as_first={r['text'] == res[0]['text']}")
            json.dump(res, open(a.out, "w"), indent=1)
dist.barrier(); sys.stdout.flush(); os._exit(0)

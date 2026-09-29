"""W=1 few-layer logic test of spec_tp on one GPU: spec K>0 must reproduce K=0 token-for-token (rows computed
bit-identically regardless of M); with oracle drafts (true continuation injected) every step must accept K;
with corrupted oracle drafts accept must stop at the corruption; prefix reuse must reproduce a fresh run."""
import os, sys, time
os.environ["MIMO_MTP"] = "1"
import torch
import mimo_tp as TP, spec_tp as SP
NLAY = int(os.environ.get("NLAY", "1"))
torch.cuda.set_device(0)
m = TP.TPModel(0, 1, layers=NLAY)
ids = list(range(1000, 1037))
with torch.inference_mode():
    sp0 = SP.Spec(m, 0); sp0.capture(1)
    ref = sp0.generate(ids, 80, stop=())
    ref2 = sp0.generate(ids + ref[:30], 30, stop=())
    idsL = [(i * 7919) % 50000 + 1000 for i in range(301)]
    refL = sp0.generate(idsL, 40, stop=())
    sp0.graphs.clear(); del sp0
    print("ref", ref[:12], flush=True)
    for K in (1, 3, 7):
        sp = SP.Spec(m, K); sp.capture(K + 1)
        if sp.PF > K + 1: sp.capture(sp.PF, pf=True)
        o = sp.generate(ids, 80, stop=())
        print(f"K={K} natural: identical {o == ref}  accepts {sp.last_accept[:12]}", flush=True)
        def oracle(s, out, bad=None):
            L = len(out)
            d = ref[L:L + K] + [0] * K
            d = d[:K]
            if bad is not None and len(d) > bad: d[bad] = 7
            s.tok[1:K + 1].copy_(torch.tensor(d))
        sp.pre_step = oracle
        o = sp.generate(ids, 80, stop=())
        print(f"K={K} oracle: identical {o == ref}  accepts {sp.last_accept[:12]}", flush=True)
        sp.pre_step = lambda s, out: oracle(s, out, bad=K // 2)
        o = sp.generate(ids, 80, stop=())
        print(f"K={K} oracle-bad@{K//2}: identical {o == ref}  accepts {sp.last_accept[:12]}", flush=True)
        sp.pre_step = None
        o = sp.generate(ids, 30, stop=())
        o2 = sp.generate(ids + ref[:30], 30, stop=(), reuse_prefix=True)
        print(f"K={K} reuse: cached {sp.cached}  identical-to-fresh {o2 == ref2}", flush=True)
        oL = sp.generate(idsL, 40, stop=())
        print(f"K={K} PF={sp.PF} 301-tok prompt: identical {oL == refL}  accepts {sp.last_accept[:8]}", flush=True)
        sp.graphs.clear(); del sp; torch.cuda.empty_cache()

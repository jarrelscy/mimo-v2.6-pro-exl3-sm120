"""L2-cold-ish MoE shard timing: graph of NT mlp_part calls on NT different tokens (different routing).
mb_moe.py LAYER [W]"""
import sys, time, itertools, torch
import mimo_fast as MF, mimo_tp as TP
M = MF.M
from concurrent.futures import ThreadPoolExecutor
L = int(sys.argv[1]); W = int(sys.argv[2]) if len(sys.argv) > 2 else 4
dev = torch.device("cuda", 0); H = MF.H; NT = 16
bb = M.Backbone()
ref = M.Layer(L, bb, dev, ThreadPoolExecutor(16))
tsc = TP.TPScratch(dev, W)
tro = {False: TP.RopeTable(MF.CFG["rope_theta"], dev, 4096), True: TP.RopeTable(MF.CFG["swa_rope_theta"], dev, 4096)}
t = TP.TPLayer(ref, 0, W, tsc, tro); del ref; torch.cuda.empty_cache()
emb = bb.get("model.embed_tokens.weight", "cpu")
torch.manual_seed(0)
X0 = (emb[torch.randint(0, 150000, (NT,))] * 8).to(dev); del emb
X = X0.clone().view(NT, 1, H)
A = torch.randn(NT, H, device=dev) * 0.5
D = torch.zeros(NT, H, device=dev)
MX = TP.MX()
def run():
    X.copy_(X0.view(NT, 1, H))
    for i in range(NT): t.mlp_part(X[i], A[i], D[i])
res = {}
with torch.inference_mode():
    cfgs = [(w, k) for w in ((0, 1), (0, 0), (1, 1), (1, 0)) for k in ((1, 1), (2, 1), (4, 1), (1, 2), (2, 2), (4, 2), (1, 4), (2, 4), (4, 4), (8, 1), (8, 2))]
    for (gu, dn), (kg, kd) in cfgs:
        MX.set_wide(gu, dn); MX.set_ksplit(kg, kd)
        run(); torch.cuda.synchronize()
        out = D.clone()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): run()
        for _ in range(3): g.replay()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(50): g.replay()
        torch.cuda.synchronize(); ms = (time.perf_counter() - t0) / 50 / NT * 1e3
        if not res: base = out
        err = ((out - base).norm() / base.norm()).item()
        res[(gu, dn, kg, kd)] = ms
        print(f"wide gu={gu} dn={dn} ksplit gu={kg} dn={kd}: mlp_part {ms*1e3:.1f} us/token  relerr-vs-first {err:.1e}", flush=True)
        del g
    b = min(res, key=res.get); print("best", b, f"{res[b]*1e3:.1f} us")

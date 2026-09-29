"""moe_cold_m (M tokens, expert-run grouping) vs per-token moe_cold. test_moe_m.py LAYER"""
import sys, time, torch
import mimo_fast as MF, mimo_tp as TP
from concurrent.futures import ThreadPoolExecutor
M_ = MF.M
L = int(sys.argv[1]); W = 4
dev = torch.device("cuda", 0); H = MF.H
bb = M_.Backbone()
ref = M_.Layer(L, bb, dev, ThreadPoolExecutor(16))
tsc = TP.TPScratch(dev, W)
tro = {False: TP.RopeTable(MF.CFG["rope_theta"], dev, 4096), True: TP.RopeTable(MF.CFG["swa_rope_theta"], dev, 4096)}
t = TP.TPLayer(ref, 0, W, tsc, tro); del ref; torch.cuda.empty_cache()
MX = TP.MX(); ir = TP.IM // W; TOPK = 8
torch.manual_seed(0)
MM = 8; S = MM * TOPK
f = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)
runs = f(2 + 2 * S + 1, dt=torch.int32); xr = f(S * 2 * H, dt=torch.half); gu_raw = f(S * 2 * ir)
act = f(S * 3 * ir, dt=torch.half); d_raw = f(S * H)
x = (torch.randn(MM, H, device=dev) * 0.3).to(torch.bfloat16)
lg = torch.randn(MM, TP.NE, device=dev)
sel = f(S, dt=torch.long); wt = f(S)
import kernels_m as KM
KM.route_m(lg, t.gate_b, sel, wt)
# force expert overlap: token1 shares 5 experts with token0, token3 = token2's experts
sv = sel.view(MM, TOPK)
sv[1, :5] = sv[0, :5]; sv[3] = sv[2]
sel = sv.reshape(-1).contiguous()
hot_y = torch.randn(S, H, device=dev) if t.has_hot else None
for M in (1, 2, 3, 4, 8):
    out = f(M, H)
    s_ = sel[:M * TOPK].contiguous(); w_ = wt[:M * TOPK].contiguous()
    MX.moe_cold_m(x[:M], s_, w_, TOPK, t.gu_ptr, t.gu_k2, t.dn_ptr, t.dn_meta, t.maxn, runs, xr, gu_raw, act, d_raw,
                  hot_y[:M * TOPK] if hot_y is not None else None, t.is_hot, None, out, 65000.0, ir)
    refo = f(M, H)
    for i in range(M):
        MX.moe_cold(x[i].contiguous(), s_[i * 8:(i + 1) * 8].contiguous(), w_[i * 8:(i + 1) * 8].contiguous(), t.gu_ptr, t.gu_k2,
                    t.dn_ptr, t.dn_meta, t.maxn, tsc.gu_raw, tsc.act, tsc.d_raw,
                    hot_y[i * 8:(i + 1) * 8].contiguous() if hot_y is not None else None, t.is_hot, None, refo[i], 65000.0, ir)
    torch.cuda.synchronize()
    d = (out - refo).abs().max().item(); rel = ((out - refo).norm() / refo.norm()).item()
    nc = int((~t.is_hot[s_]).sum().item()); nu = len(set(s_[~t.is_hot[s_]].tolist()))
    # timing (graph, 20 reps)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20):
            MX.moe_cold_m(x[:M], s_, w_, TOPK, t.gu_ptr, t.gu_k2, t.dn_ptr, t.dn_meta, t.maxn, runs, xr, gu_raw, act, d_raw,
                          hot_y[:M * TOPK] if hot_y is not None else None, t.is_hot, None, out, 65000.0, ir)
    g.replay(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10): g.replay()
    torch.cuda.synchronize(); us = (time.perf_counter() - t0) / 200 * 1e6
    print(f"L{L} M={M} cold slots {nc} unique {nu}: maxdiff {d:.2e} rel {rel:.2e}  {us:.1f} us (L2-warm)", flush=True)
g = torch.cuda.CUDAGraph()
o1 = f(H)
with torch.cuda.graph(g):
    for _ in range(20):
        MX.moe_cold(x[0].contiguous(), sel[:8], wt[:8], t.gu_ptr, t.gu_k2, t.dn_ptr, t.dn_meta, t.maxn, tsc.gu_raw, tsc.act, tsc.d_raw,
                    hot_y[:8] if hot_y is not None else None, t.is_hot, None, o1, 65000.0, ir)
g.replay(); torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(10): g.replay()
torch.cuda.synchronize(); print(f"orig moe_cold M=1: {(time.perf_counter() - t0) / 200 * 1e6:.1f} us")

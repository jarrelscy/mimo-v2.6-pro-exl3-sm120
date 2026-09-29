import sys, time, itertools, torch
sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else "60"]
exec(open("mb_moe.py").read().split("res = {}")[0])
def timeit():
    run(); torch.cuda.synchronize(); out = D.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): run()
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(40): g.replay()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / 40 / NT * 1e6, out
with torch.inference_mode():
    MX.set_wide(0, 1); MX.set_ksplit(1, 1)
    base_t, base = timeit(); print(f"baseline {base_t:.1f} us")
    best = None
    for BN, BK, nw in itertools.product((2, 4, 8, 16), (128, 256, 512, 1024), (2, 4, 8)):
        if BN * BK > 8192: continue
        TP.HOT_GU[:] = [BN, BK, nw]
        tt, o = timeit(); e = ((o - base).norm() / base.norm()).item()
        if best is None or tt < best[0]: best = (tt, BN, BK, nw)
        print(f"GU BN={BN} BK={BK} nw={nw}: {tt:.1f} us err {e:.1e}", flush=True)
    TP.HOT_GU[:] = best[1:]; print("best GU", best)
    best = None
    for BN, BK, nw in itertools.product((4, 8, 16, 32, 64), (128, 256, 512), (2, 4, 8)):
        if BN * BK > 8192: continue
        TP.HOT_DN[:] = [BN, BK, nw]
        tt, o = timeit(); e = ((o - base).norm() / base.norm()).item()
        if best is None or tt < best[0]: best = (tt, BN, BK, nw)
        print(f"DN BN={BN} BK={BK} nw={nw}: {tt:.1f} us err {e:.1e}", flush=True)
    print("best DN", best)

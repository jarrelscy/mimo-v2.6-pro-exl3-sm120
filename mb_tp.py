"""Profile rank-r shard of one layer (TP-W) on one GPU: kernel breakdown + tile-width variants."""
import sys, time, torch
import mimo_fast as MF, mimo_tp as TP
from torch.profiler import profile, ProfilerActivity
M = MF.M
from concurrent.futures import ThreadPoolExecutor
L = int(sys.argv[1]); W = int(sys.argv[2]) if len(sys.argv) > 2 else 4
dev = torch.device("cuda", 0); H = MF.H
bb = M.Backbone()
ref = M.Layer(L, bb, dev, ThreadPoolExecutor(16))
tsc = TP.TPScratch(dev, W)
tro = {False: TP.RopeTable(MF.CFG["rope_theta"], dev, 4096), True: TP.RopeTable(MF.CFG["swa_rope_theta"], dev, 4096)}
t = TP.TPLayer(ref, 0, W, tsc, tro); del ref; torch.cuda.empty_cache()
st = {"pos": torch.tensor([5], device=dev), "slot_swa": torch.tensor([5], device=dev),
      "n_full": torch.tensor([6], dtype=torch.int32, device=dev), "n_swa": torch.tensor([6], dtype=torch.int32, device=dev)}
x = (bb.get("model.embed_tokens.weight", dev)[1000:1001] * 8).clone()
a = torch.zeros(H, device=dev); d = torch.zeros(H, device=dev); z = torch.zeros(H, device=dev)
def step():
    t.attn_part(x, z, st, a); t.mlp_part(x, a, d)
from moe_ext import mod as MX
with torch.inference_mode():
    for gu, dn in ((1, 0), (0, 0), (0, 1), (1, 1)):
        MX.set_wide(gu, dn)
        for _ in range(5): step()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): step()
        for _ in range(5): g.replay()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(200): g.replay()
        torch.cuda.synchronize()
        print(f"wide gu={gu} dn={dn}: graph layer-shard {(time.perf_counter()-t0)/200*1e3:.3f} ms")
    MX.set_wide(int(sys.argv[3]) if len(sys.argv) > 3 else 0, int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    for _ in range(3): step()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10): step()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18, max_name_column_width=60))

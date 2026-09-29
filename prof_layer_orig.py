# Single-layer profile of the ORIGINAL loader on one GPU (decode T=1)
import sys, time, torch
sys.path.insert(0, "/data/Jarrel/mimo-pro-exl3-smoke")
import mimo_exl3 as M
from concurrent.futures import ThreadPoolExecutor
L = int(sys.argv[1]) if len(sys.argv) > 1 else 60
dev = torch.device("cuda", 0)
bb = M.Backbone()
lay = M.Layer(L, bb, dev, ThreadPoolExecutor(16))
torch.manual_seed(0)
x = (torch.randn(1, M.H, device=dev) * 0.05).to(torch.bfloat16)
# fill KV with some context
ctx = 64
xs = (torch.randn(ctx, M.H, device=dev) * 0.05).to(torch.bfloat16)
with torch.no_grad():
    lay.attn(M.rmsnorm(xs, lay.ln1), torch.arange(ctx, device=dev))
    def step(n):
        h = M.rmsnorm(x, lay.ln1); a = lay.attn(h, torch.tensor([n], device=dev))
        y = x + a; m = lay.mlp_fwd(M.rmsnorm(y, lay.ln2)); return y + m
    for i in range(3): step(ctx + i)
    torch.cuda.synchronize()
    def timeit(f, n=20):
        torch.cuda.synchronize(); t = time.time()
        for _ in range(n): f()
        torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
    pos = [ctx + 10]
    print("full layer ms", timeit(lambda: step(pos[0])))
    print("attn ms", timeit(lambda: lay.attn(M.rmsnorm(x, lay.ln1), torch.tensor([pos[0]], device=dev))))
    print("moe ms", timeit(lambda: lay.mlp_fwd(M.rmsnorm(x, lay.ln2))))
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for i in range(5): step(ctx + 20 + i)
        torch.cuda.synchronize()
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=25))
    ev = [e for e in p.events() if e.device_type.name == "CUDA"]
    print("cuda kernels per layer-step:", len(ev) / 5)

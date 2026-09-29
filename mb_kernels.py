import torch, time, sys
sys.path.insert(0, "/data/Jarrel/mimo-pro-exl3-smoke")
import kernels as Kn
import mimo_exl3 as M
dev = torch.device("cuda", 0)
torch.manual_seed(0)
def bench(f, n=50):
    for _ in range(3): f()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e6
bb = M.Backbone()
w = bb.get("model.layers.30.self_attn.qkv_proj.weight", dev); s = bb.get("model.layers.30.self_attn.qkv_proj.weight_scale_inv", dev)
x = (torch.randn(1, M.H, device=dev) * 0.3).to(torch.bfloat16)
ref = x @ M.qkv_dequant(w, s).T
print("ref dequant+mm us", bench(lambda: x @ M.qkv_dequant(w, s).T))
for BN in (8, 16, 32, 64):
    for nw in (2, 4, 8):
        y = Kn.fp8_gemv(x[0], w, s, M.ROWS_G, 27, BN=BN, num_warps=nw)
        err = ((y.float() - ref[0].float()).norm() / ref.float().norm()).item()
        print(f"fp8_gemv BN={BN} nw={nw} us {bench(lambda: Kn.fp8_gemv(x[0], w, s, M.ROWS_G, 27, BN=BN, num_warps=nw)):.1f} relerr {err:.2e}  GB/s {w.numel()/1e3/bench(lambda: Kn.fp8_gemv(x[0], w, s, M.ROWS_G, 27, BN=BN, num_warps=nw)):.0f}")
ow = bb.get("model.layers.30.self_attn.o_proj.weight", dev)
xo = (torch.randn(1, 16384, device=dev) * 0.3).to(torch.bfloat16)
t = bench(lambda: xo @ ow.T); print("o_proj bf16 mm us", t, "GB/s", ow.numel() * 2 / 1e3 / t)

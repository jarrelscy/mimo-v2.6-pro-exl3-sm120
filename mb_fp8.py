import torch, itertools
import kernels as Kn
dev = "cuda"; NC = 6
def bench(f, n=60):
    for _ in range(5): f(0)
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for i in range(n): f(i % NC)
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / n * 1e3
ROWS_G, SB_G = 3392, 27
N, K = 2 * ROWS_G, 6144
Wf = [torch.randn(N, K, device=dev).to(torch.float8_e4m3fn) for _ in range(NC)]
S = torch.rand(2 * SB_G, K // 128, device=dev)
xq = torch.randn(K, device=dev).bfloat16(); yq = torch.empty(N, device=dev, dtype=torch.bfloat16)
y0 = Kn.fp8_gemv(xq, Wf[0], S, ROWS_G, SB_G, BN=8).float()
print(f"old BN=8 nw=4: {bench(lambda i: Kn.fp8_gemv(xq, Wf[i], S, ROWS_G, SB_G, out=yq, BN=8, num_warps=4)):.1f} us")
for BN, BK, nw in itertools.product((4, 8, 16), (256, 512, 1024), (4, 8)):
    if BN * BK > 16384: continue
    t = bench(lambda i: Kn.fp8_gemv2(xq, Wf[i], S, ROWS_G, SB_G, out=yq, BN=BN, BK=BK, num_warps=nw))
    y = Kn.fp8_gemv2(xq, Wf[0], S, ROWS_G, SB_G, BN=BN, BK=BK, num_warps=nw).float()
    e = ((y - y0).abs().max() / y0.abs().max()).item()
    print(f"  v2 BN={BN} BK={BK} nw={nw}: {t:.1f} us  {N*K/t/1e6:.2f} TB/s  err {e:.1e}")

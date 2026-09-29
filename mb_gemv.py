"""L2-cold GEMV tuning at TP4 shard shapes (GPU3, small memory)."""
import torch, triton, itertools
import kernels as Kn
dev = "cuda"
NC = 6
def bench(f, n=60):
    for _ in range(5): f(0)
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for i in range(n): f(i % NC)
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / n * 1e3
# bf16 o_proj shard
N, K = 6144, 4096
Ws = [torch.randn(N, K, device=dev).bfloat16() for _ in range(NC)]
x = torch.randn(K, device=dev).bfloat16(); y = torch.empty(N, device=dev)
ref = Ws[0].float() @ x.float()
print(f"bf16 [{N},{K}] {N*K*2/1e6:.0f} MB")
print(f"  cublas mv bf16: {bench(lambda i: torch.mv(Ws[i], x)):.1f} us")
for BN, BK, nw in itertools.product((4, 8, 16, 32), (256, 512, 1024), (2, 4, 8)):
    if BN * BK > 16384: continue
    try:
        t = bench(lambda i: Kn.bf16_gemv(x, Ws[i], y, BN=BN, BK=BK, num_warps=nw))
    except Exception as e:
        continue
    err = ((y - ref).abs().max() / ref.abs().max()).item() if False else 0
    print(f"  triton BN={BN} BK={BK} nw={nw}: {t:.1f} us  {N*K*2/t/1e6:.2f} TB/s")
del Ws
# fp8 qkv shard
ROWS_G, SB_G = 3392, 27
N, K = 2 * ROWS_G, 6144
Wf = [torch.randn(N, K, device=dev).to(torch.float8_e4m3fn) for _ in range(NC)]
S = torch.rand(2 * SB_G, K // 128, device=dev)
xq = torch.randn(K, device=dev).bfloat16(); yq = torch.empty(N, device=dev, dtype=torch.bfloat16)
print(f"fp8 [{N},{K}] {N*K/1e6:.0f} MB")
for BN, nw in itertools.product((4, 8, 16, 32, 64), (2, 4, 8)):
    t = bench(lambda i: Kn.fp8_gemv(xq, Wf[i], S, ROWS_G, SB_G, out=yq, BN=BN, num_warps=nw))
    print(f"  fp8 BN={BN} nw={nw}: {t:.1f} us  {N*K/t/1e6:.2f} TB/s")

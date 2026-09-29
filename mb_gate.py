import torch, itertools
import kernels as Kn
NC = 24
def bench(f, n=100):
    for _ in range(5): f(0)
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for i in range(n): f(i % NC)
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / n * 1e3
Ws = [torch.randn(384, 6144, device="cuda") * 0.02 for _ in range(NC)]
h = torch.randn(1, 6144, device="cuda").bfloat16(); out = torch.empty(1, 384, device="cuda")
print(f"torch mm(h.float()): {bench(lambda i: torch.mm(h.float(), Ws[i].T, out=out)):.1f} us")
ref = torch.mm(h.float(), Ws[0].T).view(-1); y = torch.empty(384, device="cuda")
for BN, BK, nw in itertools.product((1, 2, 4), (256, 512, 1024, 2048), (2, 4, 8)):
    if BN * BK > 8192: continue
    t = bench(lambda i: Kn.gate_gemv(h.view(-1), Ws[i], y, BN, BK, nw))
    Kn.gate_gemv(h.view(-1), Ws[0], y, BN, BK, nw)
    print(f"  BN={BN} BK={BK} nw={nw}: {t:.1f} us  err {((y-ref).abs().max()/ref.abs().max()).item():.1e}")

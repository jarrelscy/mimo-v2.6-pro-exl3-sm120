import torch, kernels as K, kernels_m as KM
from mb_m import bench, NC
dev = "cuda"
N, Kd = 6784, 6144
ws = [(torch.randn(N, Kd, device=dev) * 0.05).to(torch.float8_e4m3fn) for _ in range(NC)]
s = torch.rand(8 * 7, 48, device=dev) + .5
x = torch.randn(8, Kd, device=dev).to(torch.bfloat16)
o1 = torch.empty(N, dtype=torch.bfloat16, device=dev); om = torch.empty(8, N, dtype=torch.bfloat16, device=dev)
ref = torch.empty(8, N, dtype=torch.bfloat16, device=dev)
KM.fp8_gemv_m(x, ws[0], s, 848, 7, ref, BN=4)
print("fp8 M=1 orig", round(bench(lambda c: K.fp8_gemv2(x[0], ws[c], s, 848, 7, o1)), 1))
for BN, BK, nw, st in [(16, 256, 4, 3), (16, 512, 4, 2), (16, 128, 4, 4), (32, 256, 4, 3), (16, 256, 2, 3), (32, 128, 4, 4), (16, 512, 8, 2), (64,128,4,3)]:
    r = []
    for M in (1, 4, 8):
        r.append(round(bench(lambda c: KM.fp8_gemv_md(x[:M], ws[c], s, 848, 7, om[:M], BN, BK, nw, st)), 1))
    KM.fp8_gemv_md(x, ws[0], s, 848, 7, om, BN, BK, nw, st)
    print(f"fp8 md BN={BN} BK={BK} nw={nw} st={st} M1/4/8 {r} diff {(om.float()-ref.float()).abs().max().item():.2e}")
del ws
wb = [(torch.randn(6144, 4096, device=dev) * 0.02).to(torch.bfloat16) for _ in range(NC)]
ob = torch.empty(6144, device=dev); obm = torch.empty(8, 6144, device=dev)
print("bf16 M=1 orig", round(bench(lambda c: K.bf16_gemv(x[0, :4096], wb[c], ob)), 1))
for BN, BK, nw, st in [(16, 256, 4, 3), (16, 512, 4, 2), (16, 128, 4, 4), (32, 256, 4, 3), (16, 256, 2, 3), (32, 128, 4, 4)]:
    r = []
    for M in (1, 4, 8):
        r.append(round(bench(lambda c: KM.bf16_gemv_md(x[:M, :4096], wb[c], obm[:M], BN, BK, False, nw, st)), 1))
    print(f"bf16 md BN={BN} BK={BK} nw={nw} st={st} M1/4/8 {r}")

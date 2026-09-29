"""Does an L2 prefetch of part of a weight (followed by unrelated traffic) speed up its GEMV? GPU3, small mem."""
import torch, car
import kernels as Kn
NC = 8
ROWS_G, SB_G = 3392, 27
Wf = [torch.randn(2 * ROWS_G, 6144, device="cuda").to(torch.float8_e4m3fn) for _ in range(NC)]
S = torch.rand(2 * SB_G, 48, device="cuda")
Wb = [torch.randn(6144, 4096, device="cuda").bfloat16() for _ in range(NC)]
junk = torch.empty(24 * 2**20, dtype=torch.uint8, device="cuda")   # stands in for MoE expert traffic
xq = torch.randn(6144, device="cuda").bfloat16(); yq = torch.empty(2 * ROWS_G, device="cuda", dtype=torch.bfloat16)
xb = torch.randn(4096, device="cuda").bfloat16(); yb = torch.empty(6144, device="cuda")
ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
def trial(pf_mb, which, nblk=64, thr=256, n=40, mode=1):
    tot = 0.0; tpf = 0.0
    for it in range(n):
        i = it % NC
        W = Wf[i] if which == "fp8" else Wb[i]
        ev[0].record()
        if pf_mb: car.mod.prefetch_l2(W, 0, int(pf_mb * 2**20), nblk, thr, mode)
        ev[1].record()
        junk.add_(1)  # 24 MB read+write of other data
        ev[2].record()
        if which == "fp8": Kn.fp8_gemv2(xq, W, S, ROWS_G, SB_G, out=yq, BN=8, BK=512, num_warps=8)
        else: Kn.bf16_gemv(xb, W, yb)
        ev[3].record(); torch.cuda.synchronize()
        if it >= 8: tot += ev[2].elapsed_time(ev[3]); tpf += ev[0].elapsed_time(ev[1])
    return tot / (n - 8) * 1e3, tpf / (n - 8) * 1e3
for which in ("fp8", "bf16"):
    for mb in (0, 8, 12, 16, 24, 42 if which == 'fp8' else 48):
        for mode in ((0, 1) if mb else (0,)):
          for nblk in ((32, 188) if mb else (1,)):
            g, p = trial(mb, which, nblk, mode=mode)
            print(f"{which}: mode {mode} prefetch {mb:2d} MB (blocks {nblk}): prefetch kernel {p:5.1f} us, gemv {g:5.1f} us", flush=True)

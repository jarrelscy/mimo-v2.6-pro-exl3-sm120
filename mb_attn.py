import torch, time, sys, math
sys.path.insert(0, "/data/Jarrel/mimo-pro-exl3-smoke")
import kernels as Kn
import mimo_exl3 as M
from safetensors import safe_open
dev = torch.device("cuda", 0)
torch.manual_seed(0)
def bench(f, n=50):
    for _ in range(3): f()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e6
NH, NKV, HD, VD = 128, 8, 192, 128
att = Kn.DecodeAttn(dev)
for S in (5, 100, 128, 1000, 4000, 16000):
    Lmax = max(S, 128)
    q = torch.randn(NH, HD, device=dev).to(torch.bfloat16)
    kc = torch.randn(NKV, Lmax + 64, HD, device=dev).to(torch.bfloat16) * 2
    vc = torch.randn(NKV, Lmax + 64, VD, device=dev).to(torch.bfloat16)
    sink = torch.randn(NH, device=dev).to(torch.bfloat16)
    n = torch.tensor([S], dtype=torch.int32, device=dev)
    out = torch.empty(NH, VD, dtype=torch.bfloat16, device=dev)
    for use_sink in (False, True):
        sk = sink if use_sink else None
        att(q, kc, vc, n, sk, out, HD ** -0.5)
        K = kc[:, :S].repeat_interleave(16, 0); V = vc[:, :S].repeat_interleave(16, 0)
        s = (q[:, None] @ K.transpose(1, 2)).float() * HD ** -0.5
        if use_sink: s = torch.cat((s, sink.float().view(NH, 1, 1)), -1)
        pr = torch.softmax(s, -1)
        if use_sink: pr = pr[..., :-1]
        ref = (pr.to(torch.bfloat16) @ V)[:, 0]
        err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        print(f"S={S} sink={use_sink} relerr {err:.2e} us {bench(lambda: att(q, kc, vc, n, sk, out, HD ** -0.5)):.1f}")
# NVFP4 hot
f = safe_open("/data/models/jarrelscy/MiMo-V2.6-Pro-EXL3/hot/layer60.safetensors", "pt")
hot = M.HotExperts(60, dev)
t = hot.t
slot = torch.full((384,), -1, dtype=torch.int32, device=dev)
for e, i in hot.slot.items(): slot[e] = i
hots = list(hot.slot)[:3]
idx = torch.tensor(hots + [e for e in range(384) if e not in hot.slot][:5], dtype=torch.int64, device=dev)
x = (torch.randn(8, M.H, device=dev) * 0.3).to(torch.bfloat16)
g13 = torch.zeros(8, 4096, device=dev)
Kn.nvfp4_gemv(x, slot, idx, t["w13_packed"], t["w13_bscale"], t["w13_scale2"], 0, 0, 2048, g13[:, :2048])
Kn.nvfp4_gemv(x, slot, idx, t["w13_packed"], t["w13_bscale"], t["w13_scale2"], 1, 2048, 2048, g13[:, 2048:])
for j, e in enumerate(hots):
    s_ = hot.slot[e]
    gp, up = t["w13_packed"][s_].chunk(2); gs, us = t["w13_bscale"][s_].chunk(2); g2 = t["w13_scale2"][s_]
    wg = M.nvfp4_decode(gp, gs, g2[0]); wu = M.nvfp4_decode(up, us, g2[1])
    rg = (x[j:j+1] @ wg.T).float(); ru = (x[j:j+1] @ wu.T).float()
    print("hot gate relerr", ((g13[j, :2048] - rg[0]).norm() / rg.norm()).item(), "up", ((g13[j, 2048:] - ru[0]).norm() / ru.norm()).item())
a = torch.randn(8, 2048, device=dev).to(torch.bfloat16)
d = torch.zeros(8, 6144, device=dev)
Kn.nvfp4_gemv(a, slot, idx, t["w2_packed"], t["w2_bscale"], t["w2_scale2"], 0, 0, 6144, d)
s_ = hot.slot[hots[0]]
wd = M.nvfp4_decode(t["w2_packed"][s_], t["w2_bscale"][s_], t["w2_scale2"][s_][0])
rd = (a[0:1] @ wd.T).float()
print("hot down relerr", ((d[0] - rd[0]).norm() / rd.norm()).item())
for BN, BK, nw in ((16, 256, 4), (32, 256, 4), (16, 512, 4), (32, 128, 4), (64, 256, 8), (8, 512, 4)):
    tt = bench(lambda: Kn.nvfp4_gemv(x, slot, idx, t["w13_packed"], t["w13_bscale"], t["w13_scale2"], 0, 0, 4096, g13, BN=BN, BK=BK, num_warps=nw))
    by = 3 * 4096 * 3072 * 1.0625
    print(f"nvfp4 w13 3 hot slots BN={BN} BK={BK} nw={nw} us {tt:.1f} GB/s {by/1e3/tt:.0f}")
print("ref hot expert fwd us", bench(lambda: hot.forward(hots[0], x[:1])))

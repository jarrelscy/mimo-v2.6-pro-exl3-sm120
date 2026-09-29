"""Unit tests: M-row kernels vs per-row original kernels (GPU 3, small memory)."""
import torch, math
import kernels as K
import kernels_m as KM
torch.manual_seed(0)
dev = "cuda"
M = 4
def rep(name, a, b):
    d = (a.float() - b.float()).abs().max().item()
    print(f"{name:28s} maxdiff {d:.3e} exact={d == 0}")

# fp8 qkv gemv: N=6784 (27136/4), K=6144, scale rows [216/4? use synthetic]
N, Kd = 6784, 6144
w = (torch.randn(N, Kd, device=dev) * 0.05).to(torch.float8_e4m3fn)
rows_g, sb_g = 848, 7  # synthetic grouping: ceil(848/128)=7
s = torch.rand((N // rows_g) * sb_g, Kd // 128, device=dev) + 0.5
x = torch.randn(M, Kd, device=dev).to(torch.bfloat16)
ref = torch.stack([K.fp8_gemv2(x[i], w, s, rows_g, sb_g) for i in range(M)])
out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
KM.fp8_gemv_m(x, w, s, rows_g, sb_g, out); rep("fp8_gemv_m", out, ref)

wb = (torch.randn(6144, 4096, device=dev) * 0.02).to(torch.bfloat16)
xb = torch.randn(M, 4096, device=dev).to(torch.bfloat16)
ref = torch.stack([K.bf16_gemv(xb[i], wb, torch.empty(6144, device=dev)) for i in range(M)])
out = torch.empty(M, 6144, device=dev)
KM.bf16_gemv_m(xb, wb, out); rep("bf16_gemv_m", out, ref)

wg = torch.randn(384, 6144, device=dev)
xg = torch.randn(M, 6144, device=dev).to(torch.bfloat16)
ref = torch.stack([K.gate_gemv(xg[i], wg, torch.empty(384, device=dev)) for i in range(M)])
lg = torch.empty(M, 384, device=dev)
KM.gate_gemv_m(xg, wg, lg); rep("gate_gemv_m", lg, ref)
bias = torch.randn(384, device=dev) * 0.1
idx = torch.empty(M * 8, dtype=torch.int64, device=dev); wt = torch.empty(M * 8, device=dev)
KM.route_m(lg, bias, idx, wt)
for i in range(M):
    i1 = torch.empty(8, dtype=torch.int64, device=dev); w1 = torch.empty(8, device=dev)
    K.route(lg[i].contiguous(), bias, i1, w1)
    assert torch.equal(i1, idx[i * 8:(i + 1) * 8]) and torch.equal(w1, wt[i * 8:(i + 1) * 8])
print("route_m exact")

xr = torch.randn(M, 6144, device=dev).to(torch.bfloat16); dr = torch.randn(M, 6144, device=dev)
nw = torch.randn(6144, device=dev).to(torch.bfloat16)
xo = torch.empty_like(xr); h = torch.empty_like(xr)
KM.add_rmsnorm_m(xr, nw, 1e-5, d=dr, xo=xo, h=h)
for i in range(M):
    a, b = K.add_rmsnorm(xr[i].clone(), nw, 1e-5, d=dr[i].clone())
    assert torch.equal(a, xo[i]) and torch.equal(b, h[i]), i
print("add_rmsnorm_m exact")

# qkv_post: nkv=2 per rank? use G=16, HD=192, VD=128, ROPE=64
nkv, G, HD, VD, ROPE = 2, 16, 192, 128, 64
rows = (G + 1) * HD + VD
qkv = torch.randn(M, nkv * rows, device=dev).to(torch.bfloat16)
L = 512
cos = torch.randn(L, ROPE, device=dev); sin = torch.randn(L, ROPE, device=dev)
pos = torch.tensor([100, 101, 102, 103], device=dev); slot = pos % 260
kc = torch.zeros(nkv, 260, HD, dtype=torch.bfloat16, device=dev); vc = torch.zeros(nkv, 260, VD, dtype=torch.bfloat16, device=dev)
kc2, vc2 = kc.clone(), vc.clone()
q = torch.empty(M, nkv * G, HD, dtype=torch.bfloat16, device=dev); q2 = torch.empty_like(q)
KM.qkv_post_m(qkv, cos, sin, pos, slot, q, kc, vc, 0.7, nkv, rows, G, HD, VD, ROPE)
for i in range(M):
    K.qkv_post(qkv[i].contiguous(), cos, sin, pos[i:i+1], slot[i:i+1], q2[i], kc2, vc2, 0.7, nkv, rows, G, HD, VD, ROPE)
assert torch.equal(q, q2) and torch.equal(kc, kc2) and torch.equal(vc, vc2)
print("qkv_post_m exact")

# attention full layer: rows at positions p..p+3, vs original with n = p+1
nkv = 2
Lc = 3000
kc = torch.randn(nkv, Lc, 192, device=dev).to(torch.bfloat16); vc = torch.randn(nkv, Lc, 128, device=dev).to(torch.bfloat16)
qa = torch.randn(M, nkv * 16, 192, device=dev).to(torch.bfloat16)
sink = torch.randn(nkv * 16, device=dev)
pos = torch.tensor([2000, 2001, 2002, 2003], device=dev, dtype=torch.int64)
am = KM.AttnM(dev, M, nkv); a1 = K.DecodeAttn(dev, nkv=nkv)
out = torch.empty(M, nkv * 16 * 128, dtype=torch.bfloat16, device=dev)
sc = 1 / math.sqrt(192)
am(qa, kc, vc, pos, pos[-1:], False, 0, 0, sink, out, sc)
ref = torch.stack([a1(qa[i].contiguous(), kc, vc, (pos[i:i+1] + 1).to(torch.int32), sink,
                      torch.empty(nkv * 16 * 128, dtype=torch.bfloat16, device=dev), sc) for i in range(M)])
rep("attn_m full", out, ref)

# SWA ring: R = 132, window 128; torch reference with explicit positions
def torch_attn(qv, keys, vals, sk):
    # qv [NH,192], keys [NKV,n,192]
    o = []
    for hh in range(qv.shape[0]):
        kv = hh // 16
        s_ = (keys[kv].float() @ qv[hh].float()) * sc
        s_ = torch.cat([s_, sk[hh:hh+1]])
        p_ = torch.softmax(s_, 0)[:-1]
        o.append(p_ @ vals[kv].float())
    return torch.cat(o)
R, WIN = 132, 128
for base in [3, 126, 500]:
    P = torch.arange(base, base + M, device=dev)
    kr = torch.zeros(nkv, R, 192, dtype=torch.bfloat16, device=dev); vr = torch.zeros(nkv, R, 128, dtype=torch.bfloat16, device=dev)
    kfull = torch.randn(nkv, base + M, 192, device=dev).to(torch.bfloat16); vfull = torch.randn(nkv, base + M, 128, device=dev).to(torch.bfloat16)
    # stale garbage from "future" rejected drafts beyond Pmax: place at positions Pmax+1.. (slots)
    for p in range(base + M):
        kr[:, p % R] = kfull[:, p]; vr[:, p % R] = vfull[:, p]
    am(qa, kr, vr, P, P[-1:], True, R, WIN, sink, out, sc)
    worst = 0
    for i in range(M):
        p = base + i
        lo = max(0, p - WIN + 1)
        r = torch_attn(qa[i], kfull[:, lo:p + 1], vfull[:, lo:p + 1], sink)
        worst = max(worst, (r - out[i].float()).abs().max().item())
    print(f"attn_m swa base={base:4d} vs torch maxdiff {worst:.3e}")

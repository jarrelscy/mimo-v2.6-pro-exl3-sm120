import torch, time, sys
sys.path.insert(0, "/data/Jarrel/mimo-pro-exl3-fast")
import kernels as Kn
import mimo_fast as MF
M = MF.M
dev = torch.device("cuda", 0)
torch.manual_seed(0)
H = 6144
x = torch.randn(1, H, device=dev).bfloat16(); d = torch.randn(1, H, device=dev).bfloat16(); w = (torch.rand(H, device=dev) + 0.5).bfloat16()
xo, h = Kn.add_rmsnorm(x, w, MF.EPS, d=d)
xr = x + d; hr = M.rmsnorm(xr, w)
print("add_rmsnorm diff", (xo.float() - xr.float()).abs().max().item(), (h.float() - hr.float()).abs().max().item(), hr.abs().max().item())
# qkv post
qkv = torch.randn(MF.NKV, MF.ROWS_G, device=dev).bfloat16()
rt = MF.RopeTable(MF.CFG["rope_theta"], dev, 4096)
p = 1234
pos = torch.tensor([p], device=dev); slot = torch.tensor([p % 128], device=dev)
qo = torch.empty(MF.NH, MF.HD, device=dev, dtype=torch.bfloat16)
kc = torch.zeros(MF.NKV, 128, MF.HD, device=dev, dtype=torch.bfloat16); vc = torch.zeros(MF.NKV, 128, MF.VD, device=dev, dtype=torch.bfloat16)
Kn.qkv_post(qkv, rt.cos, rt.sin, pos, slot, qo, kc, vc, MF.VSCALE, MF.NKV, MF.ROWS_G, MF.NH // MF.NKV, MF.HD, MF.VD, MF.ROPE)
q, k, v = qkv.split([MF.QG, MF.HD, MF.VD], -1)
cos, sin = rt.cos[pos], rt.sin[pos]
qr = MF.apply_rope(q.reshape(MF.NH, MF.HD), cos, sin); kr = MF.apply_rope(k, cos, sin); vr = v * MF.VSCALE
print("qkv_post q", (qo.float() - qr.float()).abs().max().item(), "k", (kc[:, p % 128].float() - kr.float()).abs().max().item(),
      "v", (vc[:, p % 128].float() - vr.float()).abs().max().item())
# route
lg = torch.randn(1, 384, device=dev); b = torch.randn(384, device=dev) * 0.1
idx = torch.empty(8, dtype=torch.long, device=dev); wt = torch.empty(8, device=dev)
Kn.route(lg, b, idx, wt)
s = lg.sigmoid(); _, ir = torch.topk(s + b[None], 8, -1); wr = s.gather(1, ir); wr = wr / (wr.sum(-1, keepdim=True) + 1e-20)
print("route idx eq", torch.equal(idx, ir[0]), "wt diff", (wt - wr[0]).abs().max().item())
def bench(f, n=200):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e6
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for _ in range(20):
        Kn.add_rmsnorm(x, w, MF.EPS, d=d, xo=xo, h=h)
print("add_rmsnorm us (graph)", bench(g.replay, 20) / 20)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for _ in range(20):
        Kn.qkv_post(qkv, rt.cos, rt.sin, pos, slot, qo, kc, vc, MF.VSCALE, MF.NKV, MF.ROWS_G, 16, MF.HD, MF.VD, MF.ROPE)
print("qkv_post us (graph)", bench(g.replay, 20) / 20)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for _ in range(20):
        Kn.route(lg, b, idx, wt)
print("route us (graph)", bench(g.replay, 20) / 20)
dq = (qo.float() - qr.float()) != 0
print("q mismatches", dq.sum().item(), "of", dq.numel(), "cols", dq.nonzero()[:, 1].unique()[:20].tolist())
i = dq.nonzero()[0]
print(qo[i[0], i[1]].item(), qr[i[0], i[1]].item())
qq = q.reshape(MF.NH, MF.HD)[i[0]]
c = i[1].item()
print("r", qq[c].item(), "cos", cos[0, c].item(), "rs", (-qq[c+32] if c < 32 else qq[c-32]).item(), "sin", sin[0, c].item())
a = (qq[c] * cos[0, c]); b = ((-qq[c+32] if c < 32 else qq[c-32]) * sin[0, c]); print("torch", a.item(), b.item(), (a+b).item())

"""Single-layer correctness + speed: FastLayer decode vs reference Layer (one GPU)."""
import sys, time, torch
import torch.nn.functional as F
import mimo_fast as MF
M = MF.M
from concurrent.futures import ThreadPoolExecutor
L = int(sys.argv[1]) if len(sys.argv) > 1 else 60
P = int(sys.argv[2]) if len(sys.argv) > 2 else 40
S = 8
dev = torch.device("cuda", 0)
bb = M.Backbone()
ref = M.Layer(L, bb, dev, ThreadPoolExecutor(16))
sc = MF.DevScratch(dev)
ropes = {False: MF.RopeTable(MF.CFG["rope_theta"], dev, MF.LMAX), True: MF.RopeTable(MF.CFG["swa_rope_theta"], dev, MF.LMAX)}
fl = MF.FastLayer(ref, sc, ropes)
print("layer", L, "swa", fl.swa, "moe", fl.moe, "hot", len(ref.hot.slot) if fl.moe else 0,
      "gate groups", [g[0] for g in fl.tab.gate] if fl.moe else None, "down groups", [(g["K"], g["maxn"], len(g["widths"])) for g in fl.tab.down] if fl.moe else None)
torch.manual_seed(0)
# realistic-ish hidden states: use embeddings of random tokens scaled
emb = bb.get("model.embed_tokens.weight", dev)
ids = torch.randint(0, 150000, (P + S,), device=dev)
X = emb[ids] * 8
st = {"pos": torch.zeros(1, dtype=torch.long, device=dev), "slot_swa": torch.zeros(1, dtype=torch.long, device=dev),
      "n_full": torch.zeros(1, dtype=torch.int32, device=dev), "n_swa": torch.zeros(1, dtype=torch.int32, device=dev)}
def setpos(p):
    st["pos"].fill_(p); st["slot_swa"].fill_(p % 128); st["n_full"].fill_(p + 1); st["n_swa"].fill_(min(p + 1, 128))
with torch.no_grad():
    ref.reset(); fl.reset()
    pos = torch.arange(P, device=dev)
    yr = ref.forward(X[:P], pos)
    yf = fl.forward_prefill(X[:P], pos)
    print("prefill relerr", ((yr.float() - yf.float()).norm() / yr.float().norm()).item())
    for i in range(S):
        p = P + i
        yr = ref.forward(X[p:p+1], torch.tensor([p], device=dev))
        setpos(p)
        yf = fl.forward_decode(X[p:p+1], st)
        # also sublayer-level: moe only
        print(f"decode pos {p} relerr {((yr.float() - yf.float()).norm() / (yr.float() - X[p:p+1].float()).norm()).item():.3e} (rel to layer delta)")
    if fl.moe:
        h = M.rmsnorm(X[P:P+1], ref.ln2)
        mr = ref.mlp_fwd(h); mf = MF.moe_decode(fl.tab, sc, h, ref.gate_w, ref.gate_b)
        print("moe-only relerr", ((mr.float() - mf.float()).norm() / mr.float().norm()).item())
        # dense fp32 reference over several random inputs (cold experts reconstructed, hot via ref)
        for trial in range(4):
            hh = M.rmsnorm(X[P - trial:P - trial + 1], ref.ln2)
            logits = hh.float() @ ref.gate_w.T; scr = logits.sigmoid()
            _, idx = torch.topk(scr + ref.gate_b[None], MF.TOPK, -1)
            wt = scr.gather(1, idx); wt = wt / wt.sum(-1, keepdim=True)
            acc = torch.zeros(1, MF.H, device=dev); ncold = 0
            for e, w in zip(idx[0].tolist(), wt[0].tolist()):
                if e in ref.cold:
                    ncold += 1
                    wg, wu, wd = ref.cold[e].weights()
                    xf = hh.float()
                    a = F.silu(xf @ wg.T) * (xf @ wu.T)
                    acc += w * (a @ wd.T)
                else:
                    acc += w * ref.hot.forward(e, hh).float()
            mr = ref.mlp_fwd(hh).float(); mf = MF.moe_decode(fl.tab, sc, hh, ref.gate_w, ref.gate_b).float()
            if hasattr(MF, "moe_decode_mgemm"):
                mg = MF.moe_decode_mgemm(fl.tab, sc, hh, ref.gate_w, ref.gate_b).float()
            else:
                mg = mf
            n = acc.norm()
            print(f"dense-fp32 check ({ncold} cold): ref relerr {((mr-acc).norm()/n).item():.3e}  fused {((mf-acc).norm()/n).item():.3e}  mgemm {((mg-acc).norm()/n).item():.3e}")
    torch.cuda.synchronize()
    def bench(f, n=30):
        for _ in range(3): f()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(n): f()
        torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
    setpos(P + S)
    x1 = X[P:P+1].clone()
    print("eager fast layer decode ms", bench(lambda: fl.forward_decode(x1, st)))
    print("eager attn ms", bench(lambda: fl.attn_decode(x1, st)))
    if fl.moe:
        print("eager moe ms", bench(lambda: MF.moe_decode(fl.tab, sc, h, ref.gate_w, ref.gate_b)))
    print("ref layer decode ms", bench(lambda: ref.forward(x1, torch.tensor([P + S], device=dev)), 10))
    ref.reset()
    # graph
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    xin = x1.clone()
    with torch.cuda.stream(s):
        for _ in range(2): out = fl.forward_decode(xin, st)
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            out = fl.forward_decode(xin, st)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    eager = fl.forward_decode(xin, st).clone()
    g.replay(); torch.cuda.synchronize()
    print("graph vs eager relerr", ((out.float() - eager.float()).norm() / eager.float().norm()).item())
    print("graph layer decode ms", bench(lambda: g.replay(), 100))
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(5): fl.forward_decode(xin, st)
        torch.cuda.synchronize()
    print(pr.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    ev = [e for e in pr.events() if e.device_type.name == "CUDA"]
    print("kernels per layer step", len(ev) / 5)

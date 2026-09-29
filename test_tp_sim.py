"""Single-GPU simulation of TP-W sharding for one layer: shards built/run one rank at a time, partials summed
by hand; compared against the (validated) PP fused FastLayer decode. Usage: test_tp_sim.py LAYER [W] [NPOS]"""
import sys, time, gc, torch
import mimo_fast as MF
import mimo_tp as TP
M = MF.M
from concurrent.futures import ThreadPoolExecutor
L = int(sys.argv[1]) if len(sys.argv) > 1 else 60
W = int(sys.argv[2]) if len(sys.argv) > 2 else 4
NP = int(sys.argv[3]) if len(sys.argv) > 3 else 12
dev = torch.device("cuda", 0)
H = MF.H
bb = M.Backbone()
ref = M.Layer(L, bb, dev, ThreadPoolExecutor(16))
torch.manual_seed(0)
emb = bb.get("model.embed_tokens.weight", "cpu")
ids = torch.randint(0, 150000, (NP,))
X = (emb[ids] * 8).to(dev); del emb


def mkst():
    return {"pos": torch.zeros(1, dtype=torch.long, device=dev), "slot_swa": torch.zeros(1, dtype=torch.long, device=dev),
            "n_full": torch.zeros(1, dtype=torch.int32, device=dev), "n_swa": torch.zeros(1, dtype=torch.int32, device=dev)}


def setpos(st, p):
    st["pos"].fill_(p); st["slot_swa"].fill_(p % 128); st["n_full"].fill_(p + 1); st["n_swa"].fill_(min(p + 1, 128))


with torch.inference_mode():
    # PP fused baseline + reference
    sc = MF.DevScratch(dev)
    ropes = {False: MF.RopeTable(MF.CFG["rope_theta"], dev, 4096), True: MF.RopeTable(MF.CFG["swa_rope_theta"], dev, 4096)}
    fl = MF.FastLayer(ref, sc, ropes)
    st = mkst(); yf = []; yr = []
    ref.reset()
    for p in range(NP):
        setpos(st, p)
        yf.append(fl.forward_decode(X[p:p + 1], st).clone())
        yr.append(ref.forward(X[p:p + 1], torch.tensor([p], device=dev)).clone())
    del fl, sc; gc.collect(); torch.cuda.empty_cache()
    tsc = TP.TPScratch(dev, W)
    tro = {False: TP.RopeTable(MF.CFG["rope_theta"], dev, 4096), True: TP.RopeTable(MF.CFG["swa_rope_theta"], dev, 4096)}
    zero = torch.zeros(H, dtype=torch.float32, device=dev)
    A = torch.zeros(NP, H, dtype=torch.float32, device=dev)
    Dm = torch.zeros(NP, H, dtype=torch.float32, device=dev)
    for phase in ("attn", "mlp"):
        for r in range(W):
            t = TP.TPLayer(ref, r, W, tsc, tro)
            st = mkst()
            for p in range(NP):
                setpos(st, p)
                x = X[p:p + 1].clone()
                if phase == "attn":
                    a = torch.zeros(H, dtype=torch.float32, device=dev)
                    t.attn_part(x, zero, st, a); A[p] += a
                else:
                    d = torch.zeros(H, dtype=torch.float32, device=dev)
                    t.mlp_part(x, A[p], d); Dm[p] += d
            if phase == "mlp" and r == 0:
                # timing of rank-0 shard parts
                x = X[:1].clone(); a = torch.zeros(H, device=dev); d = torch.zeros(H, device=dev)
                for _ in range(5): t.attn_part(x.clone(), zero, st, a); t.mlp_part(x.clone(), A[0], d)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(50): t.attn_part(x, zero, st, a)
                torch.cuda.synchronize(); t1 = time.perf_counter()
                for _ in range(50): t.mlp_part(x, A[0], d)
                torch.cuda.synchronize(); t2 = time.perf_counter()
                print(f"rank0 shard eager: attn_part {(t1-t0)/50*1e3:.3f} ms  mlp_part {(t2-t1)/50*1e3:.3f} ms")
            del t; gc.collect(); torch.cuda.empty_cache()
    for p in range(NP):
        x = X[p:p + 1].float()
        x1 = (x + A[p].bfloat16().float()).bfloat16().float()
        y = (x1 + Dm[p].bfloat16().float()).bfloat16().float()
        f, rr = yf[p].float(), yr[p].float()
        print(f"pos {p}: tp-vs-pp {((y - f).norm() / (f - x).norm()).item():.3e}  tp-vs-ref {((y - rr).norm() / (rr - x).norm()).item():.3e}"
              f"  pp-vs-ref {((f - rr).norm() / (rr - x).norm()).item():.3e}")

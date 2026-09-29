"""Kernel-time profile of one decode step: base M=1 graph and Spec K graph (5 replays each), bucketed.
torchrun --nproc-per-node 4 prof_spec.py --k 3 --out runs/prof_spec.txt"""
import argparse, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MIMO_MTP", "1")
import torch, torch.distributed as dist
from torch.profiler import profile, ProfilerActivity
from tokenizers import Tokenizer
import mimo_tp as TP, spec_tp as SP
M = TP.M
ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=3); ap.add_argument("--out", default="runs/prof_spec.txt")
a = ap.parse_args()
rank, W = TP.init_dist(); R0 = rank == 0
tok = Tokenizer.from_file(str(M.ASSETS / "tokenizer.json"))
ids = tok.encode("The capital of France is", add_special_tokens=False).ids
model = TP.TPModel(rank, W)
BUCKET = [("allreduce", ("car_kernel", "AllReduce")), ("fp8 qkv/draft gemv", ("_fp8_gemv",)), ("bf16 o_proj/lm_head gemv", ("_bf16_gemv",)),
          ("cold MoE (EXL3 trellis)", ("moe_gu", "moe_down", "moe_act", "moe_out", "moe_rot", "moe_runs")),
          ("hot MoE (NVFP4)", ("_nvfp4",)), ("gate+route", ("_gate", "_route")), ("attention", ("_attn",)),
          ("norm/rope/kv", ("rmsnorm", "_qkv_post")), ("dense L0/MTP mm", ("gemm", "gemvx", "cutlass", "Kernel2"))]
lines = []
def run(label, fn, sync):
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(5): fn(); sync()
        torch.cuda.synchronize()
    if not R0: return
    b = collections.Counter(); tot = 0.0
    for e in pr.key_averages():
        t = e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
        if t <= 0: continue
        name = next((k for k, pats in BUCKET if any(p in e.key for p in pats)), "other")
        b[name] += t / 5; tot += t / 5
    lines.append(f"== {label}: kernel time {tot/1e3:.2f} ms/step")
    for k, v in b.most_common(): lines.append(f"  {k:28s} {v/1e3:7.3f} ms  {100*v/tot:5.1f}%")
    lines.append(pr.key_averages().table(sort_by="cuda_time_total", row_limit=25))
with torch.inference_mode():
    model.capture(); model.generate(ids, 8)
    run("base M=1 graph", model.replay, lambda: model.tok_out.item())
    model.graph = None; torch.cuda.empty_cache()
    sp = SP.Spec(model, a.k); sp.capture(a.k + 1); sp.generate(ids, 8)
    run(f"spec K={a.k} (MM={a.k+1}) graph", lambda: sp.replay(a.k + 1), lambda: sp.out[0].item())
if R0:
    open(a.out, "w").write("\n".join(lines)); print("\n".join(l for l in lines if not l.startswith("-") and "|" not in l)[:6000])
dist.barrier(); sys.stdout.flush(); os._exit(0)

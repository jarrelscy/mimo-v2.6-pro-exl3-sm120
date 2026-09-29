"""Custom one-shot P2P all-reduce (car_ext.cu). Usage: car.setup(rank, W, nmax) after dist init; car.allreduce(x_fp32)."""
import os, torch, torch.distributed as dist
from torch.utils.cpp_extension import load
_d = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(_d, ".torch_ext"))
mod = load(name="mimo_car_ext", sources=[os.path.join(_d, "car_ext.cu")],
           extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17"], verbose=False)
NB = int(os.environ.get("MIMO_CAR_NB", "1"))
THREADS = int(os.environ.get("MIMO_CAR_THREADS", "0"))  # 0 = by size (car_bench_m2.py sweep)
THR1 = int(os.environ.get("MIMO_CAR_THR1", "256"))      # threads for <= 6144 floats (1 row)
THRM = int(os.environ.get("MIMO_CAR_THRM", "512"))      # threads for multi-row messages


def _thr(x):
    return THREADS if THREADS > 0 else (THR1 if x.numel() <= 6144 else THRM)


def setup(rank, W, nmax=6144, nb=NB, group=None):
    h = mod.init(rank, W, nmax, nb)
    hs = [None] * W
    dist.all_gather_object(hs, h.numpy().tobytes(), group=group)
    import numpy as np
    allh = torch.from_numpy(np.frombuffer(b"".join(hs), dtype=np.uint8).copy()).view(W, 2, -1)
    mod.open(allh)
    mod.set_fence(int(os.environ.get("MIMO_CAR_FENCE", "1")))
    dist.barrier(group=group)


PF_BLK = int(os.environ.get("MIMO_PF_BLK", "32"))


def allreduce(x, nb=-1, pft=(), pfb=(), norm=None):
    """pft/pfb: up to 3 (tensor, bytes) ranges prefetched into L2 (evict_last) by extra blocks during the AR.
    norm=(x_bf16, w_bf16, h_bf16, eps): fused epilogue x = bf16(x + bf16(sum)), h = rmsnorm(x) * w (needs nb == 1)."""
    if norm is not None:
        nb = 1
        mod.allreduce(x, _thr(x), nb, list(pft), [int(b) for b in pfb], PF_BLK if pft else 0, norm[0], norm[1], norm[2], norm[3])
    else:
        mod.allreduce(x, _thr(x), nb, list(pft), [int(b) for b in pfb], PF_BLK if pft else 0)

"""Multi-row car sweep: rows x threads x nb x fence, in-graph (140 chained ARs + 2 small kernels each).
NCCL_P2P_LEVEL=SYS torchrun --nproc-per-node 4 car_bench_m2.py"""
import os, sys, time, torch, torch.distributed as dist
rank = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
os.environ["MIMO_CAR_NB"] = "8"
import car
car.setup(rank, W, 6144 * 8)
M = car.mod
def ar(x, thr, nb): M.allreduce(x, thr, nb, [], [], 0)
# correctness for every config at 4 rows
torch.manual_seed(rank)
ok = True
for fence in (1, 0):
    M.set_fence(fence)
    for thr in (128, 256, 512, 1024):
        for nb in (1, 2, 4):
            x = torch.randn(4 * 6144, device="cuda"); gl = [torch.empty_like(x) for _ in range(W)]; dist.all_gather(gl, x)
            ref = gl[0].clone()
            for s in range(1, W): ref = ref + gl[s]
            ar(x, thr, nb); torch.cuda.synchronize()
            ok &= torch.equal(x, ref)
if rank == 0: print("all configs bit-exact:", ok, flush=True)
for rows in (1, 2, 3, 4):
    best = None
    for fence in (1, 0):
        M.set_fence(fence)
        for thr in (128, 256, 512, 1024):
            for nb in (1, 2, 4):
                if (rows * 6144) % (4 * nb): continue
                x = torch.ones(rows * 6144, device="cuda"); z = torch.ones(rows * 6144, device="cuda")
                s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3): ar(x, thr, nb)
                torch.cuda.synchronize(); dist.barrier()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(140): z.mul_(1.0001); x.copy_(z); ar(x, thr, nb)
                torch.cuda.synchronize(); dist.barrier(); g.replay(); torch.cuda.synchronize(); dist.barrier()
                t0 = time.perf_counter()
                for _ in range(20): g.replay()
                torch.cuda.synchronize(); us = (time.perf_counter() - t0) / 2800 * 1e6
                if rank == 0: print(f"rows={rows} fence={fence} thr={thr} nb={nb}: {us:.1f} us", flush=True)
                if best is None or us < best[0]: best = (us, fence, thr, nb)
                del g
    if rank == 0: print(f"BEST rows={rows}: {best}", flush=True)
M.set_fence(1)
sys.stdout.flush(); dist.barrier(); os._exit(0)

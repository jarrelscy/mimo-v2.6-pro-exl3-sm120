"""Multi-row car all-reduce latency in a graph (140 chained, with 2 small kernels between) for rows x nb.
NCCL_P2P_LEVEL=SYS MIMO_CAR_NB=8 torchrun --nproc-per-node 4 car_bench_m.py"""
import os, sys, time, torch, torch.distributed as dist
rank = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
os.environ.setdefault("MIMO_CAR_NB", "8")
import car
car.setup(rank, W, 6144 * 8)
torch.manual_seed(rank)
# correctness at 8 rows, nb=4
x = torch.randn(8 * 6144, device="cuda"); gl = [torch.empty_like(x) for _ in range(W)]; dist.all_gather(gl, x)
ref = gl[0].clone()
for s in range(1, W): ref = ref + gl[s]
car.allreduce(x, nb=4); torch.cuda.synchronize()
if rank == 0: print("8-row nb=4 bit-exact:", torch.equal(x, ref), flush=True)
for rows in (1, 2, 4, 8):
    for nb in (1, 2, 4, 8):
        x = torch.ones(rows * 6144, device="cuda"); z = torch.ones(rows * 6144, device="cuda")
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): car.allreduce(x, nb=nb)
        torch.cuda.synchronize(); dist.barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(140):
                z.mul_(1.0001); x.copy_(z); car.allreduce(x, nb=nb)
        torch.cuda.synchronize(); dist.barrier(); g.replay(); torch.cuda.synchronize(); dist.barrier()
        t0 = time.perf_counter()
        for _ in range(20): g.replay()
        torch.cuda.synchronize(); us = (time.perf_counter() - t0) / 2800 * 1e6
        if rank == 0: print(f"rows={rows} nb={nb}: {us:.1f} us/AR (incl 2 small kernels)", flush=True)
sys.stdout.flush(); dist.barrier(); os._exit(0)

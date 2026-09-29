"""All-reduce latency for small fp32 messages (decode-size), eager and inside a CUDA graph.
torchrun --nproc-per-node 4 nccl_bench.py"""
import os, time, torch, torch.distributed as dist
rank = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
for n in (6144, 2 * 6144, 65536):
    x = torch.ones(n, device="cuda")
    for _ in range(20): dist.all_reduce(x)
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    for _ in range(200): dist.all_reduce(x)
    torch.cuda.synchronize(); e = (time.perf_counter() - t0) / 200 * 1e6
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3): dist.all_reduce(x)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(140): dist.all_reduce(x)
    torch.cuda.synchronize(); dist.barrier()
    g.replay(); torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    for _ in range(10): g.replay()
    torch.cuda.synchronize(); gr = (time.perf_counter() - t0) / 1400 * 1e6
    if rank == 0:
        print(f"fp32 n={n} ({n*4/1024:.0f} KB): eager {e:.1f} us/allreduce, graph(140 chained) {gr:.1f} us/allreduce  "
              f"NCCL_P2P_LEVEL={os.environ.get('NCCL_P2P_LEVEL')} PROTO={os.environ.get('NCCL_PROTO')} ALGO={os.environ.get('NCCL_ALGO')}", flush=True)
import sys; sys.stdout.flush(); os._exit(0)

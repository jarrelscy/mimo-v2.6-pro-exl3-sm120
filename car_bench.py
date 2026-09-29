"""Custom P2P all-reduce: correctness vs NCCL + latency eager / in-graph (140 chained, with and without an
interleaved small kernel).  NCCL_P2P_LEVEL=SYS torchrun --nproc-per-node 4 car_bench.py"""
import os, sys, time, torch, torch.distributed as dist
rank = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
import car
car.setup(rank, W, 6144)
torch.manual_seed(rank)
ok = True
for it in range(50):
    x = torch.randn(6144, device="cuda") * (it + 1)
    y = x.clone(); dist.all_reduce(y)
    gl = [torch.empty_like(x) for _ in range(W)]; dist.all_gather(gl, x)
    ref = gl[0].clone()
    for s in range(1, W): ref = ref + gl[s]  # fixed order = what car does
    car.allreduce(x)
    torch.cuda.synchronize()
    if not torch.equal(x, ref): ok = False; print(f"rank{rank} it{it} MISMATCH max {(x-ref).abs().max().item()}", flush=True)
    # identical on all ranks?
    g2 = [torch.empty_like(x) for _ in range(W)]; dist.all_gather(g2, x)
    if any(not torch.equal(g2[0], g) for g in g2): ok = False; print("ranks differ", flush=True)
if rank == 0: print("correctness (bit-exact vs fixed-order sum, identical across ranks):", ok,
                   " nccl-vs-car max diff", (y - x).abs().max().item(), flush=True)


def bench(fn, label, inter=False):
    x = torch.ones(6144, device="cuda"); z = torch.ones(6144, device="cuda")
    for _ in range(20): fn(x)
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    for _ in range(500): fn(x)
    torch.cuda.synchronize(); e = (time.perf_counter() - t0) / 500 * 1e6
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn(x)
    torch.cuda.synchronize(); dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(140):
            if inter: z.mul_(1.0001); x.copy_(z)
            fn(x)
    torch.cuda.synchronize(); dist.barrier()
    g.replay(); torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    for _ in range(20): g.replay()
    torch.cuda.synchronize(); gr = (time.perf_counter() - t0) / 2800 * 1e6
    if rank == 0: print(f"{label}{' +2 small kernels' if inter else ''}: eager {e:.1f} us, graph {gr:.1f} us/allreduce", flush=True)


bench(lambda x: dist.all_reduce(x), "nccl")
bench(lambda x: dist.all_reduce(x), "nccl", True)
bench(car.allreduce, f"car nb={car.NB} thr={car.THREADS}")
bench(car.allreduce, f"car nb={car.NB} thr={car.THREADS}", True)
sys.stdout.flush(); dist.barrier(); os._exit(0)

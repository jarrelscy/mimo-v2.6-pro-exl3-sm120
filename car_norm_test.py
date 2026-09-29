"""torchrun --nproc-per-node 4 car_norm_test.py : fused AR+add+rmsnorm vs AR then Triton add_rmsnorm; AR with prefetch
blocks vs without (must be bit-identical); timing of each in a CUDA graph."""
import os, torch, torch.distributed as dist
import car, kernels as Kn
rank = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank); dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
car.setup(rank, W, 6144)
H = 6144; EPS = 1e-6
g = torch.Generator(device="cuda"); g.manual_seed(7)
x0 = (torch.randn(1, H, device="cuda", generator=g) * 3).bfloat16()
w = (1 + 0.1 * torch.randn(H, device="cuda", generator=g)).bfloat16()
torch.manual_seed(100 + rank)
part = torch.randn(H, device="cuda") * 0.5
big = torch.randn(64 * 2**20 // 4, device="cuda")
# reference: AR then triton
a = part.clone(); car.allreduce(a); xr = x0.clone(); hr = torch.empty_like(x0)
Kn.add_rmsnorm(xr, w, EPS, d=a, xo=xr, h=hr)
# prefetch AR identical
b = part.clone(); car.allreduce(b, pft=[big, big], pfb=[16 << 20, 8 << 20]); torch.cuda.synchronize()
same_pf = torch.equal(a, b)
# fused
c = part.clone(); xf = x0.clone(); hf = torch.empty_like(x0)
car.allreduce(c, norm=(xf, w, hf, EPS)); torch.cuda.synchronize()
dx = (xf.float() - xr.float()).abs().max().item(); nh = (hf != hr).sum().item()
dh = (hf.float() - hr.float()).abs().max().item()
if rank == 0:
    print(f"prefetch AR bit-identical: {same_pf}; fused x maxdiff {dx}; h mismatches {nh}/{H} (max abs {dh:.3g})", flush=True)
# timing: graph of 70 x (AR + norm) with a filler kernel between
def run(mode):
    for _ in range(70):
        big[:1 << 20].mul_(1.0)
        if mode == 0:
            car.allreduce(a); Kn.add_rmsnorm(xr, w, EPS, d=a, xo=xr, h=hr)
        else:
            car.allreduce(a, norm=(xr, w, hr, EPS))
for mode in (0, 1):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s): run(mode)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr): run(mode)
    for _ in range(5): gr.replay()
    torch.cuda.synchronize(); dist.barrier()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(50): gr.replay()
    e1.record(); torch.cuda.synchronize()
    if rank == 0: print(f"mode {'fused' if mode else 'AR+norm'}: {e0.elapsed_time(e1) / 50 / 70 * 1e3:.2f} us per (filler+AR+norm)", flush=True)
dist.barrier(); os._exit(0)

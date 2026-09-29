import torch, kernels as K, kernels_m as KM
dev = "cuda"
NC = 12
def bench(fn, iters=5):
    g = torch.cuda.CUDAGraph()
    fn(0)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for c in range(NC): fn(c)
    for _ in range(2): g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(iters):
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1) * 1000 / NC)
    return sorted(ts)[len(ts) // 2]
def _main():
    N, Kd = 6784, 6144
    ws = [(torch.randn(N, Kd, device=dev) * 0.05).to(torch.float8_e4m3fn) for _ in range(NC)]
    s = torch.rand(8 * 7, 48, device=dev) + .5
    x = torch.randn(8, Kd, device=dev).to(torch.bfloat16)
    o1 = torch.empty(N, dtype=torch.bfloat16, device=dev); om = torch.empty(8, N, dtype=torch.bfloat16, device=dev)
    print("fp8 qkv  M=1 orig", bench(lambda c: K.fp8_gemv2(x[0], ws[c], s, 848, 7, o1)))
    for M in (1, 2, 3, 4, 6, 8):
        for nw in (4, 8):
            for BN in (4, 8):
                print(f"fp8 M={M} nw={nw} BN={BN}", round(bench(lambda c: KM.fp8_gemv_m(x[:M], ws[c], s, 848, 7, om[:M], BN=BN, num_warps=nw)), 1))
    del ws
    wb = [(torch.randn(6144, 4096, device=dev) * 0.02).to(torch.bfloat16) for _ in range(NC)]
    ob = torch.empty(6144, device=dev); obm = torch.empty(8, 6144, device=dev)
    print("bf16 o M=1 orig", bench(lambda c: K.bf16_gemv(x[0, :4096], wb[c], ob)))
    for M in (1, 2, 3, 4, 6, 8):
        for nw in (4, 8):
            print(f"bf16 M={M} nw={nw}", round(bench(lambda c: KM.bf16_gemv_m(x[:M, :4096], wb[c], obm[:M], num_warps=nw)), 1))
if __name__ == "__main__":
    _main()

import sys, torch
sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else "60"]
src = open("mb_moe.py").read().split("res = {}")[0]
exec(src)
from torch.profiler import profile, ProfilerActivity
with torch.inference_mode():
    MX.set_wide(0, 1); MX.set_ksplit(1, 1)
    run(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): run()
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(10): g.replay()
        torch.cuda.synchronize()
    print(pr.key_averages().table(sort_by="cuda_time_total", row_limit=14, max_name_column_width=50))

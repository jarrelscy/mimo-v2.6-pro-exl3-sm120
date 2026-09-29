"""TP smoke + speed driver. torchrun --nproc-per-node 4 smoke_tp.py [--layers N] [--ctx 2000,4000] [--out runs/x.json]
All ranks run identical loops; rank 0 logs/writes results."""
import argparse, json, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, jinja2
import torch.distributed as dist
from tokenizers import Tokenizer
import mimo_tp as TP
M = TP.M

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=None)
ap.add_argument("--ctx", default="2000,4000")
ap.add_argument("--tests", default="completion,chat,long,ctx")
ap.add_argument("--out", default="/data/Jarrel/mimo-pro-exl3-fast/runs/smoke_tp.json")
ap.add_argument("--profile", action="store_true")
args = ap.parse_args()
rank, W = TP.init_dist()
R0 = rank == 0

tok = Tokenizer.from_file(str(M.ASSETS / "tokenizer.json"))
tmpl = jinja2.Environment().from_string(open(M.ASSETS / "chat_template.jinja").read())
enc = lambda s: tok.encode(s, add_special_tokens=False).ids
dec = lambda ids: tok.decode(ids, skip_special_tokens=False)
chat = lambda s: enc(tmpl.render(messages=[{"role": "user", "content": s}], add_generation_prompt=True, enable_thinking=False))

t0 = time.time()
model = TP.TPModel(rank, W, layers=args.layers)
load_s = time.time() - t0
if R0: M.log(f"model loaded in {load_s:.0f}s")
t0 = time.time(); model.capture()
if R0: M.log(f"capture {time.time()-t0:.1f}s")
vram = torch.cuda.memory_allocated() / 2**30

sel = args.tests.split(",")
tests = []
if "completion" in sel: tests.append(("completion", enc("The capital of France is"), 32))
if "chat" in sel: tests.append(("chat", chat("In one or two sentences, what is photosynthesis?"), 96))
if "long" in sel: tests.append(("long", chat("Write a short story (about 250 words) about a lighthouse keeper who finds a message in a bottle."), 300))
if "ctx" in sel:
    filler = open("/data/Jarrel/mimo-pro-exl3-smoke/mimo_exl3.py").read()
    filler_ids = enc(filler)
    needle = "\n# NOTE: the secret passphrase for this file is 'violet-harbor-417'.\n"
    for n in [int(c) for c in args.ctx.split(",") if c]:
        body = (filler_ids * (n // len(filler_ids) + 2))[: n - 120]
        half = len(body) // 2
        text = dec(body[:half]) + needle + dec(body[half:])
        tests.append((f"ctx{n}", chat("Here is a source file:\n\n" + text + "\n\nWhat is the secret passphrase mentioned in a NOTE comment in the file? Answer with just the passphrase."), 48))

results = {"W": W, "load_s": load_s, "vram_alloc_gib_rank0": vram,
           "env": {k: v for k, v in os.environ.items() if k.startswith(("MIMO_", "NCCL_"))}, "tests": []}
for name, ids, n in tests:
    out, t_pre, times = model.generate(ids, n, return_times=True)
    text = dec(out)
    st = {"prompt_tokens": len(ids), "new_tokens": len(out), "prefill_s": t_pre,
          "prefill_tok_s": len(ids) / t_pre}
    if times:
        ts = sorted(times)
        st.update(step_ms_median=ts[len(ts) // 2] * 1e3, step_ms_min=ts[0] * 1e3,
                  decode_tok_s=len(times) / sum(times))
    if R0:
        M.log(f"=== {name} === {st}\nOUTPUT: {text!r}")
        results["tests"].append({"name": name, "output": text, **st})
        json.dump(results, open(args.out, "w"), indent=1)
results["max_reserved_gib_rank0"] = torch.cuda.max_memory_reserved() / 2**30
if args.profile:
    from torch.profiler import profile, ProfilerActivity
    ids = enc("The capital of France is")
    model.generate(ids, 4)
    MG = os.environ.get("MIMO_GRAPHS", "1")
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as pr:
        for i in range(5):
            model.replay(); model.tok_out.item()
        torch.cuda.synchronize()
    if R0:
        tab = pr.key_averages().table(sort_by="cuda_time_total", row_limit=40)
        print(tab)
        open(args.out.replace(".json", "_prof.txt"), "w").write(tab)
        pr.export_chrome_trace(args.out.replace(".json", "_trace.json"))
if R0:
    json.dump(results, open(args.out, "w"), indent=1)
    M.log("done")
if W > 1:
    dist.barrier(); sys.stdout.flush(); os._exit(0)

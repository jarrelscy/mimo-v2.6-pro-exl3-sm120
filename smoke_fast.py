"""Fast-path smoke + speed driver (same 3 coherence tests as smoke.py, plus long-context decode speed).
python smoke_fast.py [--layers N] [--ctx 2000,4000] [--out runs/x.json]"""
import argparse, json, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, jinja2
from tokenizers import Tokenizer
import mimo_fast as MF
M = MF.M

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=None)
ap.add_argument("--ctx", default="2000,4000")
ap.add_argument("--out", default="/data/Jarrel/mimo-pro-exl3-fast/runs/smoke_fast.json")
ap.add_argument("--profile", action="store_true")
args = ap.parse_args()

tok = Tokenizer.from_file(str(M.ASSETS / "tokenizer.json"))
tmpl = jinja2.Environment().from_string(open(M.ASSETS / "chat_template.jinja").read())
enc = lambda s: tok.encode(s, add_special_tokens=False).ids
dec = lambda ids: tok.decode(ids, skip_special_tokens=False)
chat = lambda s: enc(tmpl.render(messages=[{"role": "user", "content": s}], add_generation_prompt=True, enable_thinking=False))

t0 = time.time()
model = MF.FastModel(layers=args.layers)
load_s = time.time() - t0
M.log(f"model loaded in {load_s:.0f}s; devmap {model.devmap}")
t0 = time.time(); model.capture(); M.log(f"capture {time.time()-t0:.1f}s")
vram = {i: torch.cuda.memory_allocated(i) / 2**30 for i in range(torch.cuda.device_count())}

tests = [
    ("completion", enc("The capital of France is"), 32),
    ("chat", chat("In one or two sentences, what is photosynthesis?"), 96),
    ("long", chat("Write a short story (about 250 words) about a lighthouse keeper who finds a message in a bottle."), 300),
]
filler = open("/data/Jarrel/mimo-pro-exl3-smoke/mimo_exl3.py").read()
filler_ids = enc(filler)
needle = "\n# NOTE: the secret passphrase for this file is 'violet-harbor-417'.\n"
for n in [int(c) for c in args.ctx.split(",") if c]:
    body = (filler_ids * (n // len(filler_ids) + 2))[: n - 120]
    half = len(body) // 2
    text = dec(body[:half]) + needle + dec(body[half:])
    tests.append((f"ctx{n}", chat("Here is a source file:\n\n" + text + "\n\nWhat is the secret passphrase mentioned in a NOTE comment in the file? Answer with just the passphrase."), 48))

results = {"load_s": load_s, "devmap": model.devmap, "vram_alloc_gib": vram, "env": {k: v for k, v in os.environ.items() if k.startswith("MIMO_")}, "tests": []}
for name, ids, n in tests:
    out, stats = model.generate(ids, n, return_times=True)
    text = dec(out)
    times = stats.pop("step_times")
    if times:
        ts = sorted(times)
        stats["step_ms_median"] = ts[len(ts) // 2] * 1e3
        stats["step_ms_min"] = ts[0] * 1e3
    M.log(f"=== {name} === {stats}\nOUTPUT: {text!r}")
    results["tests"].append({"name": name, "prompt_tokens": len(ids), "output": text, **stats})
    json.dump(results, open(args.out, "w"), indent=1)
results["max_reserved_gib"] = {i: torch.cuda.max_memory_reserved(i) / 2**30 for i in range(torch.cuda.device_count())}
if args.profile:
    from torch.profiler import profile, ProfilerActivity
    ids = tests[0][1]; model.reset(); lg = model.prefill(ids); nxt = lg.argmax(-1)
    for i in range(3): nxt = model.decode_step(nxt, len(ids) + i)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as pr:
        for i in range(5): nxt = model.decode_step(nxt, len(ids) + 3 + i); int(nxt)
        torch.cuda.synchronize()
    tab = pr.key_averages().table(sort_by="cuda_time_total", row_limit=40)
    print(tab)
    open(args.out.replace(".json", "_prof.txt"), "w").write(tab)
    pr.export_chrome_trace(args.out.replace(".json", "_trace.json"))
json.dump(results, open(args.out, "w"), indent=1)
M.log("done")

"""OpenAI-compatible server for MiMo-V2.6-Pro-EXL3 on the TP4 fast path.
  ./run_server.sh            (= NCCL_P2P_LEVEL=SYS MIMO_SAMPLING=1 torchrun --nproc-per-node 4 server_tp.py --port 8003)
Rank 0 serves HTTP (FastAPI/uvicorn in a thread) and broadcasts each request to all ranks over a gloo group; every
rank then runs the identical generate loop (tokens are identical on every rank, so stop decisions agree).
Requests are served one at a time (single stream). Endpoints: /v1/models, /v1/completions, /v1/chat/completions,
/health. Models: "local", "mimo-v2.6-pro-exl3" (any name is accepted)."""
import argparse, asyncio, datetime, json, os, queue, re, sys, threading, time, uuid
_d = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _d); sys.path.insert(1, os.path.join(_d, "pylib"))
import torch, jinja2
import torch.distributed as dist
from tokenizers import Tokenizer
import mimo_tp as TP
M = TP.M

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8003)
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--layers", type=int, default=None)
args = ap.parse_args()
MODEL_NAMES = ["local", "mimo-v2.6-pro-exl3"]
EOS = (151643, 151645, 151672)
DEFAULT_MAX = 4096

rank, W = TP.init_dist()
R0 = rank == 0
ctl = dist.new_group(backend="gloo", timeout=datetime.timedelta(days=30)) if W > 1 else None
tok = Tokenizer.from_file(str(M.ASSETS / "tokenizer.json"))
_jenv = jinja2.Environment(trim_blocks=False)
_jenv.filters["tojson"] = lambda v, ensure_ascii=False, indent=None, **k: json.dumps(v, ensure_ascii=ensure_ascii, indent=indent)
tmpl = _jenv.from_string(open(M.ASSETS / "chat_template.jinja").read())
enc = lambda s: tok.encode(s, add_special_tokens=False).ids
dec = lambda ids: tok.decode(ids, skip_special_tokens=False)

SPEC_K = int(os.environ.get("MIMO_SPEC", "0"))
if SPEC_K > 0:
    os.environ["MIMO_MTP"] = "1"
model = TP.TPModel(rank, W, layers=args.layers)
if SPEC_K > 0:
    # MTP self-speculation: greedy requests use the (K+1)-row verify graph, sampled requests the MM=1 graph
    import spec_tp
    gen = spec_tp.Spec(model, SPEC_K)
    gen.capture_all()
else:
    model.capture()
    gen = model
if R0: M.log(f"[server] model ready (sampling={'on' if model.sampling else 'off'}, spec K={SPEC_K}, LMAX={TP.LMAX})")


def run_job(job, emit=None):
    """Runs on every rank. emit(kind, payload) only on rank 0."""
    ids, max_new, stops = job["ids"], job["max_new"], job["stop"]
    out_ids, text_len = [], [0]
    state = {"stopped_by": None, "text": ""}
    maxstop = max((len(s) for s in stops), default=0)

    def on_token(t, i):
        if t in EOS:
            return True
        out_ids.append(t)
        if not stops and emit is None:
            return True
        text = dec(out_ids)
        if stops:
            lo = max(0, text_len[0] - maxstop)
            hit = min((p for p in (text.find(s, lo) for s in stops) if p >= 0), default=-1)
            if hit >= 0:
                state["stopped_by"] = "stop"; text = text[:hit]
                if emit: emit("text", text[text_len[0]:])
                text_len[0] = len(text); state["text"] = text
                return False
        # hold back a possible partial stop string / incomplete utf-8
        safe = len(text) - (maxstop - 1 if stops else 0)
        if text.endswith("�"): safe = min(safe, len(text) - 1)
        if emit and safe > text_len[0]:
            emit("text", text[text_len[0]:safe]); text_len[0] = safe
        state["text"] = text
        return True

    t0 = time.time()
    out = gen.generate(ids, max_new, stop=EOS, on_token=on_token, reuse_prefix=True,
                         temperature=job["temperature"], seed=job.get("seed"))
    if state["stopped_by"] is None:
        text = dec(out_ids)
        if emit and len(text) > text_len[0]: emit("text", text[text_len[0]:])
        state["text"] = text
        state["stopped_by"] = "stop" if (out and out[-1] in EOS) else "length"
    ntok = len(out)
    if emit:
        emit("done", {"text": state["text"], "finish_reason": state["stopped_by"], "prompt_tokens": len(ids),
                      "completion_tokens": ntok, "cached_tokens": gen.cached, "elapsed": time.time() - t0})


def bcast(obj):
    if W == 1: return obj
    lst = [obj]
    dist.broadcast_object_list(lst, src=0, group=ctl)
    return lst[0]


if not R0:
    while True:
        job = bcast(None)
        if job is None: continue
        if job == "exit": break
        try:
            run_job(job)
        except Exception as e:
            print(f"[rank{rank}] job error {e!r}", flush=True)
    os._exit(0)

# ------------------------------------------------------------------ rank 0: HTTP
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

jobs = queue.Queue()
app = FastAPI()


def err(msg, code=400):
    return JSONResponse({"error": {"message": msg, "type": "invalid_request_error"}}, status_code=code)


def parse_common(body, ids):
    mt = body.get("max_completion_tokens") or body.get("max_tokens")
    room = TP.LMAX - len(ids) - 1
    if room <= 0:
        raise ValueError(f"prompt too long ({len(ids)} tokens, limit {TP.LMAX})")
    max_new = min(int(mt) if mt else DEFAULT_MAX, room)
    stop = body.get("stop") or []
    if isinstance(stop, str): stop = [stop]
    temp = body.get("temperature")
    temp = 0.0 if temp is None else float(temp)
    if temp > 0 and not model.sampling: temp = 0.0
    return {"ids": ids, "max_new": max(1, max_new), "stop": [s for s in stop if s], "temperature": temp,
            "seed": body.get("seed")}


def submit(job):
    loop = asyncio.get_running_loop()
    aq = asyncio.Queue()
    job["_emit"] = lambda kind, p: loop.call_soon_threadsafe(aq.put_nowait, (kind, p))
    jobs.put(job)
    return aq


TOOL_RE = re.compile(r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.S)


def _partial_suffix(s, tag):
    """length of the longest suffix of s that is a proper prefix of tag"""
    for k in range(min(len(tag) - 1, len(s)), 0, -1):
        if tag.startswith(s[-k:]): return k
    return 0


def view(full, thinking, tools, final):
    """(reasoning, content) visible so far; non-final views hold back partial tags and tool-call bodies."""
    r, c = "", full
    if thinking:
        if not final and len(full) < 7 and "<think>".startswith(full):
            return "", ""
        if full.startswith("<think>"):
            body = full[7:]
            if "</think>" in body:
                r, c = body.split("</think>", 1)
            else:
                r, c = body, ""
                if not final: r = r[:len(r) - _partial_suffix(r, "</think>")]
    elif full.startswith("<think></think>"):
        c = full[15:]
    if tools and not final:
        k = c.find("<tool_call>")
        c = c[:k] if k >= 0 else c[:len(c) - _partial_suffix(c, "<tool_call>")]
    return r, c


def parse_tools(content):
    calls = []
    for m in TOOL_RE.finditer(content):
        name, inner = m.group(1).strip(), m.group(2)
        params = PARAM_RE.findall(inner)
        if params:
            args = {}
            for k, v in params:
                v = v.strip()
                try: args[k.strip()] = json.loads(v)
                except Exception: args[k.strip()] = v
            a = json.dumps(args, ensure_ascii=False)
        else:
            a = inner.strip() or "{}"
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function", "function": {"name": name, "arguments": a}})
    if calls:
        content = TOOL_RE.sub("", content).strip()
    return content, calls


def usage(d):
    return {"prompt_tokens": d["prompt_tokens"], "completion_tokens": d["completion_tokens"],
            "total_tokens": d["prompt_tokens"] + d["completion_tokens"],
            "prompt_tokens_details": {"cached_tokens": d["cached_tokens"]}}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    now = int(time.time())
    return {"object": "list", "data": [{"id": n, "object": "model", "created": now, "owned_by": "local",
                                        "max_model_len": TP.LMAX} for n in MODEL_NAMES]}


@app.post("/v1/completions")
async def completions(req: Request):
    body = await req.json()
    p = body.get("prompt", "")
    if isinstance(p, list):
        if p and isinstance(p[0], int): ids = list(p)
        elif len(p) == 1: ids = enc(p[0])
        else: return err("only a single prompt per request is supported")
    else:
        ids = enc(p)
    if not ids: return err("empty prompt")
    try: job = parse_common(body, ids)
    except ValueError as e: return err(str(e))
    name = body.get("model", MODEL_NAMES[0]); cid = "cmpl-" + uuid.uuid4().hex; created = int(time.time())
    aq = submit(job)
    if body.get("stream"):
        async def gen():
            while True:
                kind, pl = await aq.get()
                if kind == "text":
                    yield "data: " + json.dumps({"id": cid, "object": "text_completion", "created": created, "model": name,
                                                 "choices": [{"index": 0, "text": pl, "finish_reason": None}]}) + "\n\n"
                elif kind == "done":
                    ch = {"id": cid, "object": "text_completion", "created": created, "model": name,
                          "choices": [{"index": 0, "text": "", "finish_reason": pl["finish_reason"]}]}
                    if (body.get("stream_options") or {}).get("include_usage"): ch["usage"] = usage(pl)
                    yield "data: " + json.dumps(ch) + "\n\n"; yield "data: [DONE]\n\n"; return
                elif kind == "error":
                    yield "data: " + json.dumps({"error": {"message": pl}}) + "\n\n"; return
        return StreamingResponse(gen(), media_type="text/event-stream")
    while True:
        kind, pl = await aq.get()
        if kind == "done": break
        if kind == "error": return err(pl, 500)
    return {"id": cid, "object": "text_completion", "created": created, "model": name,
            "choices": [{"index": 0, "text": pl["text"], "finish_reason": pl["finish_reason"], "logprobs": None}],
            "usage": usage(pl)}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    msgs = body.get("messages") or []
    if not msgs: return err("messages required")
    kw = dict(body.get("chat_template_kwargs") or {})
    thinking = kw.get("enable_thinking", True) is not False
    if not thinking: kw["enable_thinking"] = False
    tools = body.get("tools") or None
    try:
        prompt = tmpl.render(messages=msgs, tools=tools, add_generation_prompt=True, **kw)
    except Exception as e:
        return err(f"chat template error: {e}")
    ids = enc(prompt)
    try: job = parse_common(body, ids)
    except ValueError as e: return err(str(e))
    name = body.get("model", MODEL_NAMES[0]); cid = "chatcmpl-" + uuid.uuid4().hex; created = int(time.time())
    aq = submit(job)

    def chunk(delta, fr=None):
        return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": fr}]}

    if body.get("stream"):
        async def gen():
            yield "data: " + json.dumps(chunk({"role": "assistant", "content": ""})) + "\n\n"
            full, sent_r, sent_c = "", 0, 0
            while True:
                kind, pl = await aq.get()
                if kind == "error":
                    yield "data: " + json.dumps({"error": {"message": pl}}) + "\n\n"; return
                final = kind == "done"
                if final: full = pl["text"]
                else: full += pl
                r, c = view(full, thinking, tools, final)
                calls = []
                if final and tools:
                    c, calls = parse_tools(c)
                if len(r) > sent_r:
                    yield "data: " + json.dumps(chunk({"reasoning_content": r[sent_r:]})) + "\n\n"; sent_r = len(r)
                if len(c) > sent_c:
                    yield "data: " + json.dumps(chunk({"content": c[sent_c:]})) + "\n\n"; sent_c = len(c)
                if final:
                    fr = pl["finish_reason"]
                    if calls:
                        yield "data: " + json.dumps(chunk({"tool_calls": [dict(index=i, **tc) for i, tc in enumerate(calls)]})) + "\n\n"
                        fr = "tool_calls"
                    last = chunk({}, fr)
                    if (body.get("stream_options") or {}).get("include_usage"): last["usage"] = usage(pl)
                    yield "data: " + json.dumps(last) + "\n\n"; yield "data: [DONE]\n\n"; return
        return StreamingResponse(gen(), media_type="text/event-stream")
    while True:
        kind, pl = await aq.get()
        if kind == "done": break
        if kind == "error": return err(pl, 500)
    r, c = view(pl["text"], thinking, tools, True)
    c, calls = parse_tools(c) if tools else (c, [])
    msg = {"role": "assistant", "content": c.strip("\n") if not calls else (c.strip() or None)}
    if r.strip(): msg["reasoning_content"] = r.strip("\n")
    fr = pl["finish_reason"]
    if calls: msg["tool_calls"] = calls; fr = "tool_calls"
    return {"id": cid, "object": "chat.completion", "created": created, "model": name,
            "choices": [{"index": 0, "message": msg, "finish_reason": fr}], "usage": usage(pl)}


def serve():
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


threading.Thread(target=serve, daemon=True).start()
M.log(f"[server] listening on {args.host}:{args.port}")
while True:
    try:
        job = jobs.get(timeout=30)
    except queue.Empty:
        bcast(None)  # heartbeat keeps the control group warm
        continue
    emit = job.pop("_emit")
    bcast(job)
    try:
        run_job(job, emit)
    except Exception as e:
        M.log(f"[server] job error {e!r}")
        emit("error", repr(e))

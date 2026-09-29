"""Smoke-test the OpenAI server on :8003."""
import json, sys, time, urllib.request
B = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8003"
def post(path, body, stream=False):
    req = urllib.request.Request(B + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=600)
    if not stream: return json.loads(r.read())
    ev = []
    for line in r:
        line = line.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]": ev.append(json.loads(line[6:]))
    return ev
print("models:", [m["id"] for m in json.loads(urllib.request.urlopen(B + "/v1/models").read())["data"]])
t = time.time(); r = post("/v1/completions", {"model": "local", "prompt": "The capital of France is", "max_tokens": 16, "temperature": 0})
print("completion:", repr(r["choices"][0]["text"]), r["choices"][0]["finish_reason"], r["usage"], f"{time.time()-t:.2f}s")
t = time.time(); r = post("/v1/chat/completions", {"model": "mimo-v2.6-pro-exl3", "messages": [{"role": "user", "content": "In one sentence, what is photosynthesis?"}],
                                             "max_tokens": 100, "chat_template_kwargs": {"enable_thinking": False}})
print("chat(no-think):", json.dumps(r["choices"][0]["message"]), r["choices"][0]["finish_reason"], r["usage"], f"{time.time()-t:.2f}s")
t = time.time(); r = post("/v1/chat/completions", {"model": "local", "messages": [{"role": "user", "content": "What is 17*23? Answer briefly."}], "max_tokens": 400})
m = r["choices"][0]["message"]
print("chat(think): reasoning", repr((m.get("reasoning_content") or "")[:200]), "... content", repr(m["content"]), r["choices"][0]["finish_reason"], r["usage"], f"{time.time()-t:.2f}s")
ev = post("/v1/chat/completions", {"model": "local", "messages": [{"role": "user", "content": "Count from 1 to 10 separated by spaces."}], "max_tokens": 60,
                                   "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}, stream=True)
txt = "".join(e["choices"][0]["delta"].get("content") or "" for e in ev if e.get("choices"))
print("stream chat:", len(ev), "events", repr(txt), ev[-1].get("usage"))
r = post("/v1/completions", {"model": "local", "prompt": "1, 2, 3, 4,", "max_tokens": 30, "stop": [" 7"]})
print("stop-string:", repr(r["choices"][0]["text"]), r["choices"][0]["finish_reason"])
r = post("/v1/completions", {"model": "local", "prompt": "Once upon a time", "max_tokens": 30, "temperature": 0.8, "seed": 1})
print("sampled t=0.8:", repr(r["choices"][0]["text"]))
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
r = post("/v1/chat/completions", {"model": "local", "messages": [{"role": "user", "content": "What's the weather in Paris right now?"}],
                                  "tools": tools, "max_tokens": 300, "chat_template_kwargs": {"enable_thinking": False}})
print("tools:", json.dumps(r["choices"][0]["message"])[:400], r["choices"][0]["finish_reason"])
# multi-turn prefix reuse
msgs = [{"role": "user", "content": "Name a colour."}]
r = post("/v1/chat/completions", {"model": "local", "messages": msgs, "max_tokens": 30, "chat_template_kwargs": {"enable_thinking": False}})
msgs += [{"role": "assistant", "content": r["choices"][0]["message"]["content"]}, {"role": "user", "content": "Another one?"}]
r = post("/v1/chat/completions", {"model": "local", "messages": msgs, "max_tokens": 30, "chat_template_kwargs": {"enable_thinking": False}})
print("turn2:", repr(r["choices"][0]["message"]["content"]), "usage", r["usage"])

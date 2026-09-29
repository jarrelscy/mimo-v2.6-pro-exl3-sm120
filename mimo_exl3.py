"""Minimal pure-PyTorch smoke-test loader for jarrelscy/MiMo-V2.6-Pro-EXL3.

Text-only MiMo-V2.6-Pro (mimo_v2, fused grouped QKV) forward:
  * backbone (attention FP8-block qkv, BF16 o_proj, router, norms, embed, lm_head) from backbone/*.safetensors
  * cold routed experts: custom mixed-rate EXL3 artifacts (selected.bin), executed with the official
    exllamav3 1.5.1 LinearEXL3 kernels (one LinearEXL3 per row-piece; pieces concatenated / scattered)
  * hot routed experts: NVFP4 (E2M1 + E4M3/16 + FP32 global), dequantized to BF16 on the fly
Layers are pipelined over the visible GPUs (naive PP, no TP). Eager attention with sinks + SWA mask,
growing KV cache. Greedy decoding. MTP / vision / audio are ignored.
"""
import json, os, sys, time, tarfile, math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from exllamav3.modules.quant.exl3 import LinearEXL3

ROOT = Path(os.environ.get("MIMO_EXL3_DIR", "/data/models/jarrelscy/MiMo-V2.6-Pro-EXL3"))
ASSETS = ROOT / "backbone" / "source_assets"
CFG = json.load(open(ASSETS / "config.json"))

H = CFG["hidden_size"]; NH = CFG["num_attention_heads"]; NKV = CFG["num_key_value_heads"]
HD = CFG["head_dim"]; VD = CFG["v_head_dim"]; ROPE = int(HD * CFG["partial_rotary_factor"])
NE = CFG["n_routed_experts"]; TOPK = CFG["num_experts_per_tok"]; EPS = CFG["layernorm_epsilon"]
NL = CFG["num_hidden_layers"]; SWA = CFG["sliding_window"]; VSCALE = CFG["attention_value_scale"]
PATTERN = CFG["hybrid_layer_pattern"]
QG = (NH // NKV) * HD  # q rows per kv group
ROWS_G = QG + HD + VD  # 3392
FP4_LUT = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- backbone
class Backbone:
    def __init__(self):
        m = json.load(open(ROOT / "backbone" / "manifest.json"))["weight_map"]
        self.map = m
        self.files = {f: safe_open(str(ROOT / "backbone" / f), "pt") for f in set(m.values())}

    def get(self, k, dev):
        return self.files[self.map[k]].get_tensor(k).to(dev)


def fp8_block_dequant(w, s, block=128):
    r, c = w.shape
    se = s.repeat_interleave(block, 0)[:r].repeat_interleave(block, 1)[:, :c]
    return (w.float() * se).to(torch.bfloat16)


def qkv_dequant(w, s):
    """Grouped layout: NKV groups of [Q(16 heads) | K | V], fp8 block scales per group (27 blocks each)."""
    wg = w.view(NKV, ROWS_G, H)
    sg = s.view(NKV, s.shape[0] // NKV, s.shape[1])
    se = sg.repeat_interleave(128, 1)[:, :ROWS_G].repeat_interleave(128, 2)[:, :, :H]
    return (wg.float() * se).to(torch.bfloat16).view(NKV * ROWS_G, H)


# ---------------------------------------------------------------- experts
def parse_artifact(raw: bytes, dev):
    meta = json.loads(raw[:4096])
    blob = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(dev)
    arrays = {}
    for spec in meta["arrays"]:
        o, n = spec["offset"], spec["bytes"]
        if spec["packed"]:
            a = np.unpackbits(np.frombuffer(raw[o:o + n], dtype=np.uint8), bitorder="little")[: int(np.prod(spec["shape"]))]
            arrays[spec["name"]] = torch.from_numpy(a.copy().reshape(spec["shape"]))
        else:
            dt = {"float16": torch.float16, "int16": torch.int16, "int32": torch.int32, "int64": torch.int64,
                  "uint8": torch.uint8, "bool": torch.bool, "float32": torch.float32}[spec["dtype"]]
            arrays[spec["name"]] = blob[o:o + n].clone().view(dt).reshape(spec["shape"])
    del blob
    assert bool(arrays["keep"].bool().all())
    return meta, arrays


class ColdExpert:
    """gate/up/down each = list of (LinearEXL3 piece, output-row index or None for contiguous)."""

    def __init__(self, raw, dev, key):
        meta, arrays = parse_artifact(raw, dev)
        self.proj = [[], [], []]
        for i, shape in enumerate(meta["projections"]):
            t = {k.split(".", 1)[1]: v for k, v in arrays.items() if k.startswith(f"{i}.")}
            j = int(t.pop("projection").item())
            blocks = t.pop("blocks", None); full_rows = t.pop("full_rows", None)
            lin = LinearEXL3(None, shape[1], shape[0], out_dtype=torch.float, key=f"{key}.{i}", **t)
            rows = None
            if blocks is not None:
                rows = (blocks.long().to(dev)[:, None] * 128 + torch.arange(128, device=dev)).flatten()
                self.full_rows = int(full_rows)
            self.proj[j].append((lin, rows, shape[0]))
        self.nbytes = len(raw)

    def _run(self, j, x, nout):
        pieces = self.proj[j]
        if pieces[0][1] is None:
            outs = [p.forward(x, {}, out_dtype=torch.float) for p, _, _ in pieces]
            return outs[0] if len(outs) == 1 else torch.cat(outs, -1)
        y = torch.empty(x.shape[0], nout, dtype=torch.float, device=x.device)
        for p, rows, _ in pieces:
            y[:, rows] = p.forward(x, {}, out_dtype=torch.float)
        return y

    def forward(self, x):  # x bf16 [T, H]
        xh = x.half().contiguous()
        g = self._run(0, xh, 2048); u = self._run(1, xh, 2048)
        a = (F.silu(g) * u).clamp(-65000, 65000).half().contiguous()
        return self._run(2, a, H).to(torch.bfloat16)

    def weights(self):
        """Reference dense reconstruction (for self-test), original [out, in] layout."""
        out = []
        for j, n in enumerate((2048, 2048, H)):
            ws = [(p.get_weight_tensor().T.float(), rows) for p, rows, _ in self.proj[j]]
            if ws[0][1] is None:
                out.append(torch.cat([w for w, _ in ws], 0))
            else:
                W = torch.empty(n, ws[0][0].shape[1], device=ws[0][0].device)
                for w, rows in ws: W[rows] = w
                out.append(W)
        return out


def nvfp4_decode(packed, scales, g):
    lut = FP4_LUT.to(packed.device)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    sc = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    return (lut[codes] * sc * g).to(torch.bfloat16)


class HotExperts:
    def __init__(self, layer, dev):
        p = ROOT / "hot" / f"layer{layer}.safetensors"
        self.t = {}
        if not p.exists():
            self.slot = {}
            return
        f = safe_open(str(p), "pt")
        pre = f"model.layers.{layer}.mlp.experts."
        kind = f.get_tensor(pre + "hyb_kind")
        hot = torch.where(kind == 0)[0].tolist()
        self.slot = {e: i for i, e in enumerate(hot)}
        for k in ("w13_packed", "w13_bscale", "w13_scale2", "w2_packed", "w2_bscale", "w2_scale2"):
            self.t[k] = f.get_tensor(pre + "nvfp4_" + k).to(dev)

    def forward(self, e, x):
        s = self.slot[e]; t = self.t
        gp, up = t["w13_packed"][s].chunk(2); gs, us = t["w13_bscale"][s].chunk(2)
        g2 = t["w13_scale2"][s]
        wg = nvfp4_decode(gp, gs, g2[0]); wu = nvfp4_decode(up, us, g2[1])
        wd = nvfp4_decode(t["w2_packed"][s], t["w2_bscale"][s], t["w2_scale2"][s][0])
        g = (x @ wg.T).float(); u = (x @ wu.T).float()
        a = (F.silu(g) * u).to(torch.bfloat16)
        return a @ wd.T


class ExpertSource:
    """Reads selected.bin either from loose files or from layers/layer_NNNNN/experts.tar (no extraction)."""

    def __init__(self, layer):
        self.layer = layer
        d = ROOT / "layers" / f"layer_{layer:05d}"
        self.tar = d / "experts.tar"
        self.index = {}
        if self.tar.exists():
            with tarfile.open(self.tar) as t:
                for m in t.getmembers():
                    if m.name.endswith("selected.bin"):
                        e = int(m.name.split("expert_")[1].split("/")[0])
                        self.index[e] = (m.offset_data, m.size)
        man = json.load(open(d / "manifest.json"))
        self.cold = sorted(x["expert"] for x in man["cold_experts"])
        self.dir = d

    def read(self, e):
        if self.index:
            off, n = self.index[e]
            with open(self.tar, "rb") as f:
                f.seek(off); return f.read(n)
        return (self.dir / f"expert_{e:05d}" / "selected.bin").read_bytes()


# ---------------------------------------------------------------- model
def rmsnorm(x, w):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return w * xf.to(x.dtype)


def rope_cos_sin(pos, theta, dev):
    inv = 1.0 / (theta ** (torch.arange(0, ROPE, 2, dtype=torch.int64, device=dev).float() / ROPE))
    fr = pos.float()[:, None] * inv[None]
    emb = torch.cat((fr, fr), -1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


def rot_half(x):
    a, b = x.chunk(2, -1)
    return torch.cat((-b, a), -1)


class Layer:
    def __init__(self, idx, bb: Backbone, dev, pool):
        self.idx, self.dev = idx, dev
        p = f"model.layers.{idx}."
        self.swa = PATTERN[idx] == 1
        self.theta = CFG["swa_rope_theta"] if self.swa else CFG["rope_theta"]
        with torch.cuda.device(dev):
            self.ln1 = bb.get(p + "input_layernorm.weight", dev)
            self.ln2 = bb.get(p + "post_attention_layernorm.weight", dev)
            self.qkv_w = bb.get(p + "self_attn.qkv_proj.weight", dev)
            self.qkv_s = bb.get(p + "self_attn.qkv_proj.weight_scale_inv", dev)
            self.o_w = bb.get(p + "self_attn.o_proj.weight", dev)
            k = p + "self_attn.attention_sink_bias"
            self.sink = bb.get(k, dev) if k in bb.map else None
            self.moe = CFG["moe_layer_freq"][idx] == 1
            if self.moe:
                self.gate_w = bb.get(p + "mlp.gate.weight", dev).float()
                self.gate_b = bb.get(p + "mlp.gate.e_score_correction_bias", dev).float()
                self.hot = HotExperts(idx, dev)
                src = ExpertSource(idx)
                self.cold = {}
                raws = pool.map(lambda e: (e, src.read(e)), src.cold)
                for e, raw in raws:
                    self.cold[e] = ColdExpert(raw, dev, f"l{idx}.e{e}")
                assert len(self.cold) + len(self.hot.slot) == NE, (idx, len(self.cold), len(self.hot.slot))
                assert not (set(self.cold) & set(self.hot.slot))
            else:
                self.mlp = [fp8_block_dequant(bb.get(p + f"mlp.{n}.weight", dev), bb.get(p + f"mlp.{n}.weight_scale_inv", dev))
                            for n in ("gate_proj", "up_proj", "down_proj")]
        self.k = self.v = None

    def reset(self):
        self.k = self.v = None

    def attn(self, x, pos):
        T = x.shape[0]
        w = qkv_dequant(self.qkv_w, self.qkv_s)
        qkv = (x @ w.T).view(T, NKV, ROWS_G)
        del w
        q, k, v = qkv.split([QG, HD, VD], -1)
        q = q.reshape(T, NH, HD).transpose(0, 1)  # [NH,T,HD] head h = g*16+i
        k = k.transpose(0, 1)  # [NKV,T,HD]
        v = (v * VSCALE).transpose(0, 1)
        cos, sin = rope_cos_sin(pos, self.theta, x.device)
        def ap(t):
            r, n = t[..., :ROPE], t[..., ROPE:]
            return torch.cat((r * cos + rot_half(r) * sin, n), -1)
        q, k = ap(q), ap(k)
        if self.k is None:
            self.k, self.v, self.kpos = k, v, pos
        else:
            self.k = torch.cat((self.k, k), 1); self.v = torch.cat((self.v, v), 1); self.kpos = torch.cat((self.kpos, pos))
        K = self.k.repeat_interleave(NH // NKV, 0); V = self.v.repeat_interleave(NH // NKV, 0)
        s = (q @ K.transpose(1, 2)).float() * (HD ** -0.5)  # [NH,T,S]
        d = pos[:, None] - self.kpos[None, :]
        mask = d < 0
        if self.swa: mask = mask | (d >= SWA)
        s = s.masked_fill(mask[None], float("-inf"))
        if self.sink is not None:
            s = torch.cat((s, self.sink.float().view(NH, 1, 1).expand(NH, T, 1)), -1)
        pr = torch.softmax(s, -1)
        if self.sink is not None: pr = pr[..., :-1]
        o = (pr.to(torch.bfloat16) @ V).transpose(0, 1).reshape(T, NH * VD)
        return o @ self.o_w.T

    def mlp_fwd(self, x):
        if not self.moe:
            g, u, d = self.mlp
            return (F.silu(x @ g.T) * (x @ u.T)) @ d.T
        logits = x.float() @ self.gate_w.T
        sc = logits.sigmoid()
        _, idx = torch.topk(sc + self.gate_b[None], TOPK, -1)
        wt = sc.gather(1, idx); wt = wt / (wt.sum(-1, keepdim=True) + 1e-20)
        out = torch.zeros(x.shape[0], H, dtype=torch.float, device=x.device)
        idx_l = idx.tolist()
        by_e = {}
        for t, row in enumerate(idx_l):
            for s, e in enumerate(row): by_e.setdefault(e, []).append((t, s))
        for e, ts in by_e.items():
            tt = torch.tensor([a for a, _ in ts], device=x.device); ss = torch.tensor([b for _, b in ts], device=x.device)
            xe = x[tt].contiguous()
            ye = self.cold[e].forward(xe) if e in self.cold else self.hot.forward(e, xe)
            out.index_add_(0, tt, ye.float() * wt[tt, ss][:, None])
        return out.to(torch.bfloat16)

    def forward(self, x, pos):
        with torch.cuda.device(self.dev):
            x = x + self.attn(rmsnorm(x, self.ln1), pos)
            x = x + self.mlp_fwd(rmsnorm(x, self.ln2))
            return x


def layer_bytes():
    out = [0.6e9]
    for l in range(1, NL):
        man = json.load(open(ROOT / "layers" / f"layer_{l:05d}" / "manifest.json"))
        c = sum(e["bytes"] for e in man["cold_experts"])
        hp = ROOT / "hot" / f"layer{l}.safetensors"
        out.append(c + (hp.stat().st_size if hp.exists() else 0) + 0.37e9)
    return out


def assign(ngpu, extra_first=1.9e9, extra_last=1.9e9):
    b = layer_bytes(); tot = sum(b) + extra_first + extra_last
    target = tot / ngpu; dev = []; acc = extra_first; g = 0
    for l, x in enumerate(b):
        remaining = NL - l
        if g < ngpu - 1 and acc + x / 2 > target * (g + 1) and remaining > 0:
            g += 1
        dev.append(g); acc += x
    return dev


class Model:
    def __init__(self, ngpu=None, layers=None, threads=16):
        ngpu = ngpu or torch.cuda.device_count()
        self.devmap = assign(ngpu)
        nl = layers or NL
        bb = Backbone()
        d0, dl = torch.device("cuda", 0), torch.device("cuda", self.devmap[nl - 1])
        self.embed = bb.get("model.embed_tokens.weight", d0)
        self.norm = bb.get("model.norm.weight", dl)
        self.lm_head = bb.get("lm_head.weight", dl)
        self.layers = []
        pool = ThreadPoolExecutor(threads)
        t0 = time.time()
        for l in range(nl):
            dev = torch.device("cuda", self.devmap[l])
            self.layers.append(Layer(l, bb, dev, pool))
            if l % 5 == 0 or l == nl - 1:
                log(f"loaded layer {l} on {dev} ({time.time()-t0:.0f}s) mem " +
                    " ".join(f"{torch.cuda.memory_allocated(i)/2**30:.1f}" for i in range(ngpu)))
        self.dl = dl

    def reset(self):
        for L in self.layers: L.reset()

    @torch.no_grad()
    def forward(self, ids, pos):
        x = self.embed[ids.to(self.embed.device)]
        for L in self.layers:
            x = L.forward(x.to(L.dev), pos.to(L.dev))
        x = rmsnorm(x.to(self.dl), self.norm)
        return (x[-1:] @ self.lm_head.T).float()

    @torch.no_grad()
    def generate(self, ids, max_new, stop=(151643, 151645, 151672)):
        self.reset()
        ids = torch.tensor(ids, dtype=torch.long)
        pos = torch.arange(len(ids))
        torch.cuda.synchronize(); t0 = time.time()
        logits = self.forward(ids, pos)
        torch.cuda.synchronize(); tp = time.time() - t0
        out = []
        t1 = time.time()
        n = len(ids)
        for i in range(max_new):
            nxt = int(logits.argmax(-1))
            out.append(nxt)
            if nxt in stop: break
            logits = self.forward(torch.tensor([nxt]), torch.tensor([n])); n += 1
        torch.cuda.synchronize(); td = time.time() - t1
        return out, {"prompt_tokens": len(ids), "prefill_s": tp, "decode_tokens": len(out),
                     "decode_s": td, "decode_tok_s": (len(out) - 1) / td if len(out) > 1 else 0.0}

"""Fast batch-1 decode for MiMo-V2.6-Pro-EXL3.

Built on top of the reference loader (/data/Jarrel/mimo-pro-exl3-smoke/mimo_exl3.py, left untouched):
weights are loaded by the reference Layer class, then decode-side structures are added:
  * FP8 block-scaled qkv via a Triton GEMV (no per-step dequant)
  * static KV caches (full layers: [NKV, LMAX]; SWA layers: 128-slot ring) + Triton split-KV decode attention w/ sinks
  * routed experts: device-side routing, cold EXL3 experts via exllamav3 exl3_mgemm with per-layer pointer
    tables (one launch per (projection, bitrate[, piece-ordinal]) group, null pointer = skip), hot NVFP4
    experts via a Triton GEMV that reads the expert slot table on device
  * no host syncs inside a decode step -> each pipeline stage (GPU) is captured as one CUDA graph
Prefill (T>1) runs the reference eager math (chunked), writing into the static caches.
"""
import os, sys, time, math
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/Jarrel/mimo-pro-exl3-smoke")
import mimo_exl3 as M  # noqa: E402
from mimo_exl3 import H, NH, NKV, HD, VD, ROPE, NE, TOPK, EPS, NL, SWA, VSCALE, QG, ROWS_G, CFG, log  # noqa
from exllamav3.ext import exllamav3_ext as ext  # noqa: E402
import kernels as Kn  # noqa: E402

LMAX = int(os.environ.get("MIMO_LMAX", 32768))
SB_G = (ROWS_G + 127) // 128  # 27 fp8 scale row-blocks per kv group
MOE_FORCE_SHAPE = int(os.environ.get("MIMO_MOE_SHAPE", 0))  # 0 = autotune for uniform-width groups


def rmsnorm(x, w):
    return M.rmsnorm(x, w)


class RopeTable:
    def __init__(self, theta, dev, lmax):
        pos = torch.arange(lmax, device=dev)
        self.cos, self.sin = M.rope_cos_sin(pos, theta, dev)  # [lmax, ROPE] bf16


def apply_rope(t, cos, sin):
    r, n = t[..., :ROPE], t[..., ROPE:]
    return torch.cat((r * cos + M.rot_half(r) * sin, n), -1)


# ------------------------------------------------------------------------------------------- MoE tables
class MoETables:
    """exl3_mgemm pointer tables for the cold experts of one layer + NVFP4 hot slot table."""

    def __init__(self, layer: "M.Layer", dev):
        self.dev = dev
        z = lambda: [0] * NE
        gate, up, down = {}, {}, {}
        self.keep = []  # keep tensors alive
        for e, ce in layer.cold.items():
            for j, grp in ((0, gate), (1, up)):
                pieces = ce.proj[j]
                assert len(pieces) == 1 and pieces[0][1] is None and pieces[0][2] == 2048
                lin = pieces[0][0]
                assert lin.mul1 and not lin.mcg
                t = grp.setdefault(lin.K, {"B": z(), "suh": z(), "svh": z()})
                t["B"][e] = lin.trellis.data_ptr(); t["suh"][e] = lin.suh.data_ptr(); t["svh"][e] = lin.svh.data_ptr()
            col = 0; ordk = {}
            for lin, rows, nout in ce.proj[2]:
                assert rows is None and lin.mul1 and not lin.mcg
                o = ordk.get(lin.K, 0); ordk[lin.K] = o + 1
                t = down.setdefault((lin.K, o), {"B": z(), "suh": z(), "svh": z(), "n": z(), "col": z(), "maxn": 0})
                t["B"][e] = lin.trellis.data_ptr(); t["suh"][e] = lin.suh.data_ptr(); t["svh"][e] = lin.svh.data_ptr()
                t["n"][e] = nout; t["col"][e] = col; t["maxn"] = max(t["maxn"], nout)
                col += nout
            assert col == H, (e, col)
        # fused heterogeneous-K tables for moe_ext (see moe_ext.cu)
        gp = torch.zeros(NE, 2, 3, dtype=torch.long); gk = torch.zeros(NE, 2, dtype=torch.int32)
        dp = torch.zeros(NE, 3, 3, dtype=torch.long); dm = torch.zeros(NE, 3, 3, dtype=torch.int32)
        maxn = 0
        for e, ce in layer.cold.items():
            for j in range(2):
                lin = ce.proj[j][0][0]
                gp[e, j] = torch.tensor([lin.trellis.data_ptr(), lin.suh.data_ptr(), lin.svh.data_ptr()])
                gk[e, j] = int(round(lin.K * 2))
            col = 0
            assert len(ce.proj[2]) <= 3
            for p, (lin, rows, nout) in enumerate(ce.proj[2]):
                dp[e, p] = torch.tensor([lin.trellis.data_ptr(), lin.suh.data_ptr(), lin.svh.data_ptr()])
                dm[e, p] = torch.tensor([int(round(lin.K * 2)), nout, col])
                assert nout % 128 == 0 and col % 128 == 0
                col += nout; maxn = max(maxn, nout)
        assert set(gk.flatten().tolist()) | set(dm[:, :, 0].flatten().tolist()) <= {0, 3, 4, 5, 6}, "unsupported K"
        self.gu_ptr, self.gu_k2, self.dn_ptr, self.dn_meta = gp.to(dev), gk.to(dev), dp.to(dev), dm.to(dev)
        self.maxn = maxn
        L = lambda v: torch.tensor(v, dtype=torch.long, device=dev)
        self.gate = [(K, L(t["B"]), L(t["suh"]), L(t["svh"])) for K, t in sorted(gate.items())]
        self.up = [(K, L(t["B"]), L(t["suh"]), L(t["svh"])) for K, t in sorted(up.items())]
        self.down = []
        for (K, o), t in sorted(down.items()):
            self.down.append(dict(K=K, B=L(t["B"]), suh=L(t["suh"]), svh=L(t["svh"]),
                                  n=torch.tensor(t["n"], dtype=torch.int32, device=dev),
                                  col=torch.tensor(t["col"], dtype=torch.long, device=dev),
                                  maxn=t["maxn"], widths=sorted(set(x for x in t["n"] if x))))
        self.cptr = None
        hot = layer.hot
        self.has_hot = len(hot.slot) > 0
        slot = [-1] * NE
        for e, i in hot.slot.items(): slot[e] = i
        self.hot_slot = torch.tensor(slot, dtype=torch.int32, device=dev)
        is_hot = [s >= 0 for s in slot]
        self.is_hot = torch.tensor(is_hot, dtype=torch.bool, device=dev)
        self.ht = hot.t

    def bind_scratch(self, D):
        """Output pointers for the down groups into the per-device scratch D [NE, H] fp32."""
        base = D.data_ptr()
        for g in self.down:
            g["cptr"] = base + (torch.arange(NE, device=self.dev, dtype=torch.long) * H + g["col"]) * 4
            g["cptr"] = torch.where(g["n"] > 0, g["cptr"], torch.zeros_like(g["cptr"]))


class DevScratch:
    """Per-device scratch shared by all layers on a device (layers run sequentially)."""

    def __init__(self, dev):
        self.dev = dev
        self.G = torch.zeros(TOPK, 1, 2048, dtype=torch.float32, device=dev)
        self.U = torch.zeros(TOPK, 1, 2048, dtype=torch.float32, device=dev)
        self.D = torch.zeros(NE, H, dtype=torch.float32, device=dev)
        self.Ahad = torch.zeros(NE * H, dtype=torch.half, device=dev)
        self.Hgu = torch.zeros(TOPK, 4096, dtype=torch.float32, device=dev)
        self.Hd = torch.zeros(TOPK, H, dtype=torch.float32, device=dev)
        self.Cdummy = {}
        self.gu_raw = torch.zeros(TOPK * 2 * 2048, dtype=torch.float32, device=dev)
        self.act = torch.zeros(TOPK * 3 * 2048, dtype=torch.half, device=dev)
        self.d_raw = torch.zeros(TOPK * H, dtype=torch.float32, device=dev)
        self.moe_out = torch.zeros(H, dtype=torch.bfloat16, device=dev)
        self.logits = torch.zeros(1, NE, dtype=torch.float32, device=dev)
        self.sel = torch.zeros(TOPK, dtype=torch.long, device=dev)
        self.wt = torch.zeros(TOPK, dtype=torch.float32, device=dev)
        self.h = torch.zeros(1, H, dtype=torch.bfloat16, device=dev)
        self.x2 = torch.zeros(1, H, dtype=torch.bfloat16, device=dev)
        self.q = torch.zeros(NH, HD, dtype=torch.bfloat16, device=dev)
        self.xbuf = [torch.zeros(1, H, dtype=torch.bfloat16, device=dev) for _ in range(2)]
        self.attn = Kn.DecodeAttn(dev)
        self.qkv = torch.empty(NKV * ROWS_G, dtype=torch.bfloat16, device=dev)
        self.attn_out = torch.empty(NH, VD, dtype=torch.bfloat16, device=dev)

    def cdummy(self, n):
        if n not in self.Cdummy:
            self.Cdummy[n] = torch.empty(1, 1, n, dtype=torch.float32, device=self.dev)
        return self.Cdummy[n]


def pick_shape(widths):
    if MOE_FORCE_SHAPE:
        return MOE_FORCE_SHAPE
    if all(w % 512 == 0 for w in widths):
        return -1
    if all(w % 256 == 0 for w in widths):
        return 3
    return 2


MOE_IMPL = os.environ.get("MIMO_MOE_IMPL", "fused")


def moe_decode(tab, sc, x, gate_w, gate_b):
    if MOE_IMPL == "mgemm":
        return moe_decode_mgemm(tab, sc, x, gate_w, gate_b)
    return moe_decode_fused(tab, sc, x, gate_w, gate_b)


def moe_decode_fused(tab: MoETables, sc: DevScratch, x, gate_w, gate_b):
    """x [1, H] bf16 -> [1, H] bf16 via moe_ext (4 launches for cold experts) + Triton NVFP4 hot experts."""
    from moe_ext import mod as MX
    logits = x.float() @ gate_w.T
    s = logits.sigmoid()
    _, idx = torch.topk(s + gate_b[None], TOPK, -1)
    wt = s.gather(1, idx); wt = wt / (wt.sum(-1, keepdim=True) + 1e-20)
    i0 = idx[0]
    hot_y = None
    if tab.has_hot:
        t = tab.ht
        xb = x.expand(TOPK, H)
        Kn.nvfp4_gemv(xb, tab.hot_slot, i0, t["w13_packed"], t["w13_bscale"], t["w13_scale2"], 0, 0, 2048, sc.Hgu[:, :2048])
        Kn.nvfp4_gemv(xb, tab.hot_slot, i0, t["w13_packed"], t["w13_bscale"], t["w13_scale2"], 1, 2048, 2048, sc.Hgu[:, 2048:])
        ha = (F.silu(sc.Hgu[:, :2048]) * sc.Hgu[:, 2048:]).to(torch.bfloat16)
        Kn.nvfp4_gemv(ha, tab.hot_slot, i0, t["w2_packed"], t["w2_bscale"], t["w2_scale2"], 0, 0, H, sc.Hd)
        hot_y = sc.Hd
    MX.moe_cold(x.view(H), i0, wt[0], tab.gu_ptr, tab.gu_k2, tab.dn_ptr, tab.dn_meta, tab.maxn,
                sc.gu_raw, sc.act, sc.d_raw, hot_y, tab.is_hot, None, sc.moe_out, 65000.0, 2048)
    return sc.moe_out.view(1, H)


FUSED = os.environ.get("MIMO_FUSED", "1") == "1"


def moe_decode_fused2(tab: MoETables, sc: DevScratch, h, res, gate_w, gate_b, out):
    """Fully fused decode MoE: out = res + moe(h). h/res/out [1, H] bf16."""
    from moe_ext import mod as MX
    logits = torch.mm(h.float(), gate_w.T, out=sc.logits)
    Kn.route(logits, gate_b, sc.sel, sc.wt, TOPK)
    hot_y = None
    if tab.has_hot:
        t = tab.ht
        xb = h.expand(TOPK, H)
        Kn.nvfp4_gemv(xb, tab.hot_slot, sc.sel, t["w13_packed"], t["w13_bscale"], t["w13_scale2"], 0, 0, 2048, sc.Hgu[:, :2048])
        Kn.nvfp4_gemv(xb, tab.hot_slot, sc.sel, t["w13_packed"], t["w13_bscale"], t["w13_scale2"], 1, 2048, 2048, sc.Hgu[:, 2048:])
        Kn.nvfp4_gemv(sc.Hgu, tab.hot_slot, sc.sel, t["w2_packed"], t["w2_bscale"], t["w2_scale2"], 0, 0, H, sc.Hd, act_off=2048)
        hot_y = sc.Hd
    MX.moe_cold(h.view(H), sc.sel, sc.wt, tab.gu_ptr, tab.gu_k2, tab.dn_ptr, tab.dn_meta, tab.maxn,
                sc.gu_raw, sc.act, sc.d_raw, hot_y, tab.is_hot, res.view(H), out.view(H), 65000.0, 2048)
    return out


def moe_decode_mgemm(tab: MoETables, sc: DevScratch, x, gate_w, gate_b):
    """x [1, H] bf16 -> [1, H] bf16. Graph-safe (no host syncs)."""
    logits = x.float() @ gate_w.T
    s = logits.sigmoid()
    _, idx = torch.topk(s + gate_b[None], TOPK, -1)  # [1, 8] int64
    wt = s.gather(1, idx); wt = wt / (wt.sum(-1, keepdim=True) + 1e-20)
    xh = x.half().view(1, 1, H)
    for K, B, suh, svh in tab.gate:
        ext.exl3_mgemm(xh, B, sc.G, suh, sc.Ahad, svh, idx, None, K, -1, False, True, -1, -1, 0, 1, None, None)
    for K, B, suh, svh in tab.up:
        ext.exl3_mgemm(xh, B, sc.U, suh, sc.Ahad, svh, idx, None, K, -1, False, True, -1, -1, 0, 1, None, None)
    a = (F.silu(sc.G) * sc.U).clamp(-65000, 65000).half()  # [8, 1, 2048]
    for g in tab.down:
        ext.exl3_mgemm(a, g["B"], sc.cdummy(g["maxn"]), g["suh"], sc.Ahad, g["svh"], idx, None, g["K"],
                       pick_shape(g["widths"]), False, True, -1, -1, 0, 1, g["n"], g["cptr"])
    y = sc.D[idx[0]]  # [8, H] fp32
    if tab.has_hot:
        t = tab.ht; i0 = idx[0]
        xb = x.expand(TOPK, H)
        Kn.nvfp4_gemv(xb, tab.hot_slot, i0, t["w13_packed"], t["w13_bscale"], t["w13_scale2"], 0, 0, 2048, sc.Hgu[:, :2048])
        Kn.nvfp4_gemv(xb, tab.hot_slot, i0, t["w13_packed"], t["w13_bscale"], t["w13_scale2"], 1, 2048, 2048, sc.Hgu[:, 2048:])
        ha = (F.silu(sc.Hgu[:, :2048]) * sc.Hgu[:, 2048:]).to(torch.bfloat16)
        Kn.nvfp4_gemv(ha, tab.hot_slot, i0, t["w2_packed"], t["w2_bscale"], t["w2_scale2"], 0, 0, H, sc.Hd)
        # reference: hot expert output is bf16 (a @ wd.T in bf16)
        y = torch.where(tab.is_hot[i0][:, None], sc.Hd.to(torch.bfloat16).float(), y)
    # reference: cold output cast to bf16 before the fp32 weighted accumulation
    y = y.to(torch.bfloat16).float()
    return (y * wt[0][:, None]).sum(0, keepdim=True).to(torch.bfloat16)


# ------------------------------------------------------------------------------------------- layer
class FastLayer:
    def __init__(self, ref: "M.Layer", scratch: DevScratch, ropes):
        self.r = ref
        self.idx, self.dev, self.swa = ref.idx, ref.dev, ref.swa
        self.sc = scratch
        self.rope = ropes[self.swa]
        L = SWA if self.swa else LMAX
        self.kc = torch.zeros(NKV, L, HD, dtype=torch.bfloat16, device=self.dev)
        self.vc = torch.zeros(NKV, L, VD, dtype=torch.bfloat16, device=self.dev)
        # positions of cached keys (for prefill masks); ring for SWA
        self.kpos = torch.full((L,), -10 ** 9, dtype=torch.long, device=self.dev)
        self.moe = ref.moe
        if self.moe:
            self.tab = MoETables(ref, self.dev)
            self.tab.bind_scratch(scratch.D)

    def reset(self):
        self.kpos.fill_(-10 ** 9)

    # ---- decode (T=1), graph-safe; st = per-device step state
    def attn_decode(self, x, st):
        r, sc = self.r, self.sc
        h = rmsnorm(x, r.ln1)
        qkv = Kn.fp8_gemv(h[0], r.qkv_w, r.qkv_s, ROWS_G, SB_G, out=sc.qkv).view(NKV, ROWS_G)
        q, k, v = qkv.split([QG, HD, VD], -1)
        q = q.reshape(NH, HD)
        v = v * VSCALE
        cos = self.rope.cos[st["pos"]]; sin = self.rope.sin[st["pos"]]  # [1, ROPE]
        q = apply_rope(q, cos, sin).contiguous()
        k = apply_rope(k, cos, sin)
        slot = st["slot_swa"] if self.swa else st["pos"]
        self.kc.index_copy_(1, slot, k[:, None])
        self.vc.index_copy_(1, slot, v[:, None])
        n = st["n_swa"] if self.swa else st["n_full"]
        o = sc.attn(q, self.kc, self.vc, n, r.sink, sc.attn_out, HD ** -0.5)
        return o.view(1, NH * VD) @ r.o_w.T

    def forward_decode(self, x, st, out=None):
        if FUSED:
            return self.forward_decode_fused(x, st, out)
        return self.forward_decode_unfused(x, st)

    def forward_decode_fused(self, x, st, out=None):
        """x [1, H] bf16 -> new residual [1, H] (written to out if given). ~16 kernels/layer."""
        r, sc = self.r, self.sc
        _, h = Kn.add_rmsnorm(x, r.ln1, EPS, h=sc.h)
        qkv = Kn.fp8_gemv(h[0], r.qkv_w, r.qkv_s, ROWS_G, SB_G, out=sc.qkv)
        slot = st["slot_swa"] if self.swa else st["pos"]
        Kn.qkv_post(qkv, self.rope.cos, self.rope.sin, st["pos"], slot, sc.q, self.kc, self.vc, VSCALE,
                    NKV, ROWS_G, NH // NKV, HD, VD, ROPE)
        n = st["n_swa"] if self.swa else st["n_full"]
        o = sc.attn(sc.q, self.kc, self.vc, n, r.sink, sc.attn_out, HD ** -0.5)
        a = o.view(1, NH * VD) @ r.o_w.T
        x2, h2 = Kn.add_rmsnorm(x, r.ln2, EPS, d=a, xo=sc.x2, h=sc.h)
        if out is None:
            out = torch.empty_like(x)
        if self.moe:
            return moe_decode_fused2(self.tab, sc, h2, x2, r.gate_w, r.gate_b, out)
        g, u, d = r.mlp
        out.copy_(x2 + (F.silu(h2 @ g.T) * (h2 @ u.T)) @ d.T)
        return out

    def forward_decode_unfused(self, x, st):
        x = x + self.attn_decode(x, st)
        r = self.r
        h = rmsnorm(x, r.ln2)
        if self.moe:
            m = moe_decode(self.tab, self.sc, h, r.gate_w, r.gate_b)
        else:
            g, u, d = r.mlp
            m = (F.silu(h @ g.T) * (h @ u.T)) @ d.T
        return x + m

    # ---- prefill (T>=1), eager reference math with static caches
    def attn_prefill(self, x, pos):
        r = self.r
        T = x.shape[0]
        w = M.qkv_dequant(r.qkv_w, r.qkv_s)
        qkv = (x @ w.T).view(T, NKV, ROWS_G)
        del w
        q, k, v = qkv.split([QG, HD, VD], -1)
        q = q.reshape(T, NH, HD).transpose(0, 1)
        k = k.transpose(0, 1)
        v = (v * VSCALE).transpose(0, 1)
        cos, sin = self.rope.cos[pos], self.rope.sin[pos]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.swa:
            K = torch.cat((self.kc, k), 1); V = torch.cat((self.vc, v), 1); kp = torch.cat((self.kpos, pos))
            # write last <=SWA new tokens into the ring
            tail = slice(max(0, T - SWA), T)
            sl = pos[tail] % SWA
            self.kc.index_copy_(1, sl, k[:, tail]); self.vc.index_copy_(1, sl, v[:, tail]); self.kpos.index_copy_(0, sl, pos[tail])
        else:
            self.kc.index_copy_(1, pos, k); self.vc.index_copy_(1, pos, v); self.kpos.index_copy_(0, pos, pos)
            n = int(pos[-1]) + 1
            K, V, kp = self.kc[:, :n], self.vc[:, :n], self.kpos[:n]
        out = torch.empty(T, NH * VD, dtype=torch.bfloat16, device=x.device)
        CH = 256
        for c0 in range(0, T, CH):
            c1 = min(T, c0 + CH)
            qc = q[:, c0:c1]
            Kr = K.repeat_interleave(NH // NKV, 0); Vr = V.repeat_interleave(NH // NKV, 0)
            s = (qc @ Kr.transpose(1, 2)).float() * (HD ** -0.5)
            d = pos[c0:c1, None] - kp[None, :]
            mask = d < 0
            if self.swa: mask = mask | (d >= SWA)
            s = s.masked_fill(mask[None], float("-inf"))
            if r.sink is not None:
                s = torch.cat((s, r.sink.float().view(NH, 1, 1).expand(NH, c1 - c0, 1)), -1)
            pr = torch.softmax(s, -1)
            if r.sink is not None: pr = pr[..., :-1]
            out[c0:c1] = (pr.to(torch.bfloat16) @ Vr).transpose(0, 1).reshape(c1 - c0, NH * VD)
            del Kr, Vr, s, pr
        return out @ r.o_w.T

    def forward_prefill(self, x, pos):
        x = x + self.attn_prefill(M.rmsnorm(x, self.r.ln1), pos)
        return x + self.r.mlp_fwd(M.rmsnorm(x, self.r.ln2))


# ------------------------------------------------------------------------------------------- model
class FastModel:
    def __init__(self, ngpu=None, layers=None, threads=16, devmap=None):
        ngpu = ngpu or torch.cuda.device_count()
        self.devmap = devmap or M.assign(ngpu)
        nl = layers or NL
        bb = M.Backbone()
        d0, dl = torch.device("cuda", self.devmap[0]), torch.device("cuda", self.devmap[nl - 1])
        self.embed = bb.get("model.embed_tokens.weight", d0)
        self.norm = bb.get("model.norm.weight", dl)
        self.lm_head = bb.get("lm_head.weight", dl)
        self.devs = sorted(set(self.devmap[:nl]))
        self.scratch = {d: DevScratch(torch.device("cuda", d)) for d in self.devs}
        self.ropes = {d: {False: RopeTable(CFG["rope_theta"], torch.device("cuda", d), LMAX),
                          True: RopeTable(CFG["swa_rope_theta"], torch.device("cuda", d), LMAX)} for d in self.devs}
        self.layers = []
        pool = ThreadPoolExecutor(threads)
        t0 = time.time()
        for l in range(nl):
            d = self.devmap[l]; dev = torch.device("cuda", d)
            ref = M.Layer(l, bb, dev, pool)
            with torch.cuda.device(dev):
                self.layers.append(FastLayer(ref, self.scratch[d], self.ropes[d]))
            if l % 10 == 0 or l == nl - 1:
                log(f"loaded layer {l} on {dev} ({time.time()-t0:.0f}s) mem " +
                    " ".join(f"{torch.cuda.memory_allocated(i)/2**30:.1f}" for i in range(torch.cuda.device_count())))
        self.dl = dl
        self.stages = []  # (dev, [layers])
        for L in self.layers:
            if not self.stages or self.stages[-1][0] != L.dev:
                self.stages.append((L.dev, []))
            self.stages[-1][1].append(L)
        # per-device step state
        self.st = {}
        for d in self.devs:
            dev = torch.device("cuda", d)
            self.st[d] = {"pos": torch.zeros(1, dtype=torch.long, device=dev),
                          "slot_swa": torch.zeros(1, dtype=torch.long, device=dev),
                          "n_full": torch.zeros(1, dtype=torch.int32, device=dev),
                          "n_swa": torch.zeros(1, dtype=torch.int32, device=dev)}
        self.pos_host = torch.zeros(4, dtype=torch.long).pin_memory()
        self.tok_in = torch.zeros(1, dtype=torch.long, device=d0)
        self.stage_in = [torch.zeros(1, H, dtype=torch.bfloat16, device=dv) for dv, _ in self.stages]
        self.stage_out = [torch.zeros(1, H, dtype=torch.bfloat16, device=dv) for dv, _ in self.stages]
        self.tok_out = torch.zeros(1, dtype=torch.long, device=dl)
        self.logits_out = torch.zeros(1, self.lm_head.shape[0], dtype=torch.float32, device=dl)
        self.graphs = None
        self.use_graphs = os.environ.get("MIMO_GRAPHS", "1") == "1"

    def reset(self):
        for L in self.layers: L.reset()

    # ---- step state
    def set_pos(self, p):
        h = self.pos_host
        h[0] = p; h[1] = p % SWA; h[2] = p + 1; h[3] = min(p + 1, SWA)
        for d, st in self.st.items():
            dev = torch.device("cuda", d)
            v = h.to(dev, non_blocking=True)
            st["pos"].copy_(v[0:1]); st["slot_swa"].copy_(v[1:2])
            st["n_full"].copy_(v[2:3]); st["n_swa"].copy_(v[3:4])

    def _stage_fn(self, si):
        dev, layers = self.stages[si]
        st = self.st[dev.index]

        def f():
            if si == 0:
                x = self.embed[self.tok_in].view(1, H)
            else:
                x = self.stage_in[si]
            for li, L in enumerate(layers):
                x = L.forward_decode(x, st, out=L.sc.xbuf[li % 2])
            if si == len(self.stages) - 1:
                h = rmsnorm(x, self.norm)
                lg = (h @ self.lm_head.T).float()
                self.logits_out.copy_(lg)
                self.tok_out.copy_(lg.argmax(-1))
            else:
                self.stage_out[si].copy_(x)
        return f

    def _run_stages(self):
        for si, (dev, _) in enumerate(self.stages):
            with torch.cuda.device(dev):
                if si > 0:
                    self.stage_in[si].copy_(self.stage_out[si - 1])
                if self.graphs is not None:
                    self.graphs[si].replay()
                else:
                    self.fns[si]()

    def capture(self):
        self.fns = [self._stage_fn(i) for i in range(len(self.stages))]
        # warmup (autotune exl3 kernels, compile triton) eagerly
        for _ in range(2):
            self._run_stages()
        torch.cuda.synchronize()
        if not self.use_graphs:
            return
        self.graphs = []
        self.pools = []
        for si, (dev, _) in enumerate(self.stages):
            with torch.cuda.device(dev):
                s = torch.cuda.Stream(dev)
                s.wait_stream(torch.cuda.current_stream(dev))
                g = torch.cuda.CUDAGraph()
                with torch.cuda.stream(s):
                    with torch.cuda.graph(g, stream=s):
                        self.fns[si]()
                torch.cuda.current_stream(dev).wait_stream(s)
                self.graphs.append(g)
        torch.cuda.synchronize()

    @torch.no_grad()
    def decode_step(self, tok_dev, p):
        """tok_dev: long [1] on any device (next input token); p: its position. Returns tok_out (device)."""
        self.set_pos(p)
        self.tok_in.copy_(tok_dev.view(1))
        self._run_stages()
        return self.tok_out

    @torch.no_grad()
    def prefill(self, ids, p0=0, chunk=512):
        """ids: list/1D long of prompt tokens at positions p0.. ; returns logits of last token."""
        ids = torch.as_tensor(ids, dtype=torch.long)
        for c0 in range(0, len(ids), chunk):
            cid = ids[c0:c0 + chunk]
            pos = torch.arange(p0 + c0, p0 + c0 + len(cid))
            x = self.embed[cid.to(self.embed.device)]
            for L in self.layers:
                with torch.cuda.device(L.dev):
                    x = L.forward_prefill(x.to(L.dev), pos.to(L.dev))
        x = rmsnorm(x[-1:].to(self.dl), self.norm)
        return (x @ self.lm_head.T).float()

    @torch.no_grad()
    def generate(self, ids, max_new, stop=(151643, 151645, 151672), return_times=False):
        self.reset()
        if self.graphs is None and not hasattr(self, "fns"):
            self.capture()
            self.reset()
        torch.cuda.synchronize(); t0 = time.time()
        logits = self.prefill(ids)
        torch.cuda.synchronize(); tp = time.time() - t0
        out = []
        n = len(ids)
        nxt = logits.argmax(-1)
        t1 = time.time(); times = []
        for i in range(max_new):
            tk = int(nxt)
            out.append(tk)
            if tk in stop or i == max_new - 1: break
            ts = time.time()
            nxt = self.decode_step(nxt, n); n += 1
            if return_times:
                torch.cuda.synchronize(); times.append(time.time() - ts)
        torch.cuda.synchronize(); td = time.time() - t1
        st = {"prompt_tokens": len(ids), "prefill_s": tp, "decode_tokens": len(out),
              "decode_s": td, "decode_tok_s": (len(out) - 1) / td if len(out) > 1 else 0.0,
              "ms_per_step": td / max(1, len(out) - 1) * 1e3}
        if return_times: st["step_times"] = times
        return out, st

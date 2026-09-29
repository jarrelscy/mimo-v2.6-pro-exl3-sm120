"""Tensor-parallel (TP=W, default 4) batch-1 decode for MiMo-V2.6-Pro-EXL3. One process per GPU, NCCL.

Sharding per rank r (all splits are exact: the EXL3 Hadamards are 128-blockwise):
  attention : KV groups [r*NKV/W, (r+1)*NKV/W) -> fp8 qkv rows + per-group scales, sinks, KV cache;
              o_proj input columns [r*4096, (r+1)*4096) -> fp32 partial
  experts   : intermediate slice [r*I/W, (r+1)*I/W) of every expert:
              gate/up trellis n-tiles + svh slice (suh full); down trellis k-tiles + suh slice (svh full);
              hot NVFP4 w13 rows / w2 packed columns
  dense L0  : intermediate slice of gate/up rows, down columns
  lm_head   : vocab rows slice -> local argmax -> all_gather
Residual stream: X bf16 [H] + pending fp32 delta D [H] (the all-reduced partial sum); every layer starts with
X = bf16(X + bf16(D)), h = rmsnorm(X) in one kernel -> exactly the reference residual rounding.
Per layer: 2 fp32 all-reduces (24 KB).
"""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
import torch.distributed as dist

sys.path.insert(0, "/data/Jarrel/mimo-pro-exl3-smoke")
import mimo_exl3 as M  # noqa: E402
from mimo_exl3 import H, NH, NKV, HD, VD, ROPE, NE, TOPK, EPS, NL, SWA, VSCALE, ROWS_G, CFG, log  # noqa
import kernels as Kn  # noqa: E402

LMAX = int(os.environ.get("MIMO_LMAX", 32768))
FP8V2 = os.environ.get("MIMO_FP8V2", "1") == "1"
GATEK = os.environ.get("MIMO_GATEK", "1") == "1"
# L2 prefetch (evict_last) by extra blocks of the custom all-reduce. Spec "A=q42,g9/D=o24": A = attention AR (before the
# MoE), D = MoE AR (end of layer); q = next layer qkv, o = next layer o_proj, g = router weight (A: this layer, D: next).
def parse_pf(spec):
    out = {"A": [], "D": []}
    for part in filter(None, spec.split("/")):
        k, v = part.split("=")
        out[k] = [(t[0], float(t[1:])) for t in v.split(",") if t]
    return out
PF = parse_pf(os.environ.get("MIMO_PF", ""))
# fuse residual add + next rmsnorm into the custom all-reduce epilogue
FUSENORM = os.environ.get("MIMO_FUSENORM", "0") == "1"
KSMAX = 8
HOT_GU = [2, 1024, 4]  # nvfp4 hot gate/up (BN, BK, num_warps)
HOT_DN = [16, 256, 4]  # nvfp4 hot down
SB_G = (ROWS_G + 127) // 128
IM = 2048  # moe intermediate
ID = CFG["intermediate_size"]  # dense intermediate


def cp(t):
    """Fresh contiguous copy (never a view of the full-layer storage, so the reference layer can be freed)."""
    return torch.empty(t.shape, dtype=t.dtype, device=t.device).copy_(t)


def MX():
    from moe_ext import mod
    return mod


class RopeTable:
    def __init__(self, theta, dev, lmax):
        pos = torch.arange(lmax, device=dev)
        self.cos, self.sin = M.rope_cos_sin(pos, theta, dev)


class TPScratch:
    def __init__(self, dev, W):
        self.W = W
        f = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)
        ir = IM // W
        self.gu_raw = f(KSMAX * TOPK * 2 * ir); self.act = f(TOPK * 3 * ir, dt=torch.half); self.d_raw = f(KSMAX * TOPK * H)
        self.Hgu = f(TOPK, 2 * ir); self.Hd = f(TOPK, H)
        self.logits = f(1, NE); self.sel = f(TOPK, dt=torch.long); self.wt = f(TOPK)
        self.h = f(1, H, dt=torch.bfloat16)
        self.qkv = f(NKV // W * ROWS_G, dt=torch.bfloat16)
        self.q = f(NH // W, HD, dt=torch.bfloat16)
        self.attn_out = f(NH // W, VD, dt=torch.bfloat16)
        self.attn = Kn.DecodeAttn(dev, nkv=NKV // W)
        self.dgu = f(2 * ID // W, dt=torch.bfloat16)


class TPLayer:
    """One layer's shard for rank r. Built from a full reference M.Layer (which the caller then drops)."""

    def __init__(self, ref, r, W, sc: TPScratch, ropes):
        self.idx, self.swa, self.moe, self.dev = ref.idx, ref.swa, ref.moe, ref.dev
        self.sc, self.rope = sc, ropes[self.swa]
        c = cp
        nkv = NKV // W; nh = NH // W
        self.nkv = nkv
        self.ln1, self.ln2 = cp(ref.ln1), cp(ref.ln2)
        self.qkv_w = c(ref.qkv_w[r * nkv * ROWS_G:(r + 1) * nkv * ROWS_G])
        sbg = ref.qkv_s.shape[0] // NKV
        assert sbg == SB_G
        self.qkv_s = c(ref.qkv_s[r * nkv * SB_G:(r + 1) * nkv * SB_G])
        self.o_w = c(ref.o_w[:, r * nh * VD:(r + 1) * nh * VD])
        self.sink = c(ref.sink[r * nh:(r + 1) * nh]) if ref.sink is not None else None
        L = SWA if self.swa else LMAX
        self.kc = torch.zeros(nkv, L, HD, dtype=torch.bfloat16, device=self.dev)
        self.vc = torch.zeros(nkv, L, VD, dtype=torch.bfloat16, device=self.dev)
        if self.moe:
            self.gate_w, self.gate_b = cp(ref.gate_w), cp(ref.gate_b)
            self._shard_moe(ref, r, W)
        else:
            g, u, d = ref.mlp
            ir = ID // W
            self.gu_w = torch.cat((g[r * ir:(r + 1) * ir], u[r * ir:(r + 1) * ir]), 0)
            self.d_w = c(d[:, r * ir:(r + 1) * ir])

    def _shard_moe(self, ref, r, W):
        dev = self.dev
        ir = IM // W
        keep = []
        gp = torch.zeros(NE, 2, 3, dtype=torch.long); gk = torch.zeros(NE, 2, dtype=torch.int32)
        dp = torch.zeros(NE, 3, 3, dtype=torch.long); dm = torch.zeros(NE, 3, 3, dtype=torch.int32)
        maxn = 0
        P = lambda *ts: torch.tensor([t.data_ptr() for t in ts])
        for e, ce in ref.cold.items():
            for j in range(2):
                pieces = ce.proj[j]
                assert len(pieces) == 1 and pieces[0][1] is None and pieces[0][2] == IM
                lin = pieces[0][0]
                assert lin.mul1 and not lin.mcg
                nt = lin.trellis.shape[1] // W
                tr = cp(lin.trellis[:, r * nt:(r + 1) * nt])
                svh = cp(lin.svh[r * ir:(r + 1) * ir])
                suh = cp(lin.suh)
                keep += [tr, svh, suh]
                gp[e, j] = P(tr, suh, svh); gk[e, j] = int(round(lin.K * 2))
            col = 0
            assert len(ce.proj[2]) <= 3
            for p, (lin, rows, nout) in enumerate(ce.proj[2]):
                assert rows is None and lin.mul1 and not lin.mcg
                kt = lin.trellis.shape[0] // W
                tr = cp(lin.trellis[r * kt:(r + 1) * kt])
                suh = cp(lin.suh[r * ir:(r + 1) * ir])
                svh = cp(lin.svh)
                keep += [tr, suh, svh]
                dp[e, p] = P(tr, suh, svh)
                dm[e, p] = torch.tensor([int(round(lin.K * 2)), nout, col])
                assert nout % 128 == 0 and col % 128 == 0
                col += nout; maxn = max(maxn, nout)
            assert col == H
        assert set(gk.flatten().tolist()) | set(dm[:, :, 0].flatten().tolist()) <= {0, 3, 4, 5, 6}
        self.keep = keep
        self.gu_ptr, self.gu_k2, self.dn_ptr, self.dn_meta = gp.to(dev), gk.to(dev), dp.to(dev), dm.to(dev)
        self.maxn = maxn
        hot = ref.hot
        self.has_hot = len(hot.slot) > 0
        slot = [-1] * NE
        for e, i in hot.slot.items(): slot[e] = i
        self.hot_slot = torch.tensor(slot, dtype=torch.int32, device=dev)
        self.is_hot = torch.tensor([s >= 0 for s in slot], dtype=torch.bool, device=dev)
        if self.has_hot:
            t = hot.t
            sl = lambda a: torch.cat((a[:, r * ir:(r + 1) * ir], a[:, IM + r * ir:IM + (r + 1) * ir]), 1)
            self.ht = {"w13_packed": sl(t["w13_packed"]), "w13_bscale": sl(t["w13_bscale"]),
                       "w13_scale2": cp(t["w13_scale2"]),
                       "w2_packed": cp(t["w2_packed"][:, :, r * ir // 2:(r + 1) * ir // 2]),
                       "w2_bscale": cp(t["w2_bscale"][:, :, r * ir // 16:(r + 1) * ir // 16]),
                       "w2_scale2": cp(t["w2_scale2"])}

    # ---- decode phases (graph-safe). X bf16 [1,H] updated in place, D fp32 [H] pending delta.
    def attn_part(self, X, D, st, a_out, normed=False):
        sc = self.sc
        h = sc.h if normed else Kn.add_rmsnorm(X, self.ln1, EPS, d=D, xo=X, h=sc.h)[1]
        qkv = (Kn.fp8_gemv2(h[0], self.qkv_w, self.qkv_s, ROWS_G, SB_G, out=sc.qkv, BN=8, BK=512, num_warps=8) if FP8V2 else Kn.fp8_gemv(h[0], self.qkv_w, self.qkv_s, ROWS_G, SB_G, out=sc.qkv))
        slot = st["slot_swa"] if self.swa else st["pos"]
        Kn.qkv_post(qkv, self.rope.cos, self.rope.sin, st["pos"], slot, sc.q, self.kc, self.vc, VSCALE,
                    self.nkv, ROWS_G, NH // NKV, HD, VD, ROPE)
        n = st["n_swa"] if self.swa else st["n_full"]
        o = sc.attn(sc.q, self.kc, self.vc, n, self.sink, sc.attn_out, HD ** -0.5)
        Kn.bf16_gemv(o.view(-1), self.o_w, a_out)
        return a_out

    def mlp_part(self, X, A, d_out, normed=False):
        """X = bf16(X + bf16(A)); d_out = partial mlp(rmsnorm(X)) fp32."""
        sc = self.sc
        h = sc.h if normed else Kn.add_rmsnorm(X, self.ln2, EPS, d=A, xo=X, h=sc.h)[1]
        if not self.moe:
            gu = torch.mm(h, self.gu_w.T)  # bf16 like the reference
            ir = self.gu_w.shape[0] // 2
            a = F.silu(gu[:, :ir]) * gu[:, ir:]
            Kn.bf16_gemv(a.view(-1), self.d_w, d_out)
            return d_out
        W = self.sc.W; ir = IM // W
        if GATEK:
            Kn.gate_gemv(h.view(-1), self.gate_w, sc.logits.view(-1), 1, 1024, 8); logits = sc.logits
        else:
            logits = torch.mm(h.float(), self.gate_w.T, out=sc.logits)
        Kn.route(logits, self.gate_b, sc.sel, sc.wt, TOPK)
        hot_y = None
        if self.has_hot:
            t = self.ht
            xb = h.expand(TOPK, H)
            Kn.nvfp4_gemv(xb, self.hot_slot, sc.sel, t["w13_packed"], t["w13_bscale"], t["w13_scale2"], 0, 0, ir, sc.Hgu[:, :ir],
                          BN=HOT_GU[0], BK=HOT_GU[1], num_warps=HOT_GU[2])
            Kn.nvfp4_gemv(xb, self.hot_slot, sc.sel, t["w13_packed"], t["w13_bscale"], t["w13_scale2"], 1, ir, ir, sc.Hgu[:, ir:],
                          BN=HOT_GU[0], BK=HOT_GU[1], num_warps=HOT_GU[2])
            Kn.nvfp4_gemv(sc.Hgu, self.hot_slot, sc.sel, t["w2_packed"], t["w2_bscale"], t["w2_scale2"], 0, 0, H, sc.Hd,
                          act_off=ir, BN=HOT_DN[0], BK=HOT_DN[1], num_warps=HOT_DN[2])
            hot_y = sc.Hd
        MX().moe_cold(h.view(H), sc.sel, sc.wt, self.gu_ptr, self.gu_k2, self.dn_ptr, self.dn_meta, self.maxn,
                      sc.gu_raw, sc.act, sc.d_raw, hot_y, self.is_hot, None, d_out, 65000.0, ir)
        return d_out


class _MTPRef:
    """Duck-typed M.Layer for MTP layer j (SWA attention with sinks + dense fp8 MLP) so TPLayer can shard it."""

    def __init__(self, j, bb, dev):
        p = f"model.mtp.layers.{j}."
        self.idx, self.swa, self.moe, self.dev = -1 - j, True, False, dev
        self.ln1 = bb.get(p + "input_layernorm.weight", dev)
        self.ln2 = bb.get(p + "pre_mlp_layernorm.weight", dev)
        self.qkv_w = bb.get(p + "self_attn.qkv_proj.weight", dev)
        self.qkv_s = bb.get(p + "self_attn.qkv_proj.weight_scale_inv", dev)
        self.o_w = bb.get(p + "self_attn.o_proj.weight", dev)
        self.sink = bb.get(p + "self_attn.attention_sink_bias", dev)
        self.mlp = [M.fp8_block_dequant(bb.get(p + f"mlp.{n}.weight", dev), bb.get(p + f"mlp.{n}.weight_scale_inv", dev))
                    for n in ("gate_proj", "up_proj", "down_proj")]
        self.enorm = bb.get(p + "enorm.weight", dev)
        self.hnorm = bb.get(p + "hnorm.weight", dev)
        self.fnorm = bb.get(p + "final_layernorm.weight", dev)
        self.eh = bb.get(p + "eh_proj.weight", dev)  # [H, 2H] bf16, input = cat(enorm(emb), hnorm(h))


class TPMTP:
    """MTP draft layer shard: h' = eh_proj(cat(enorm(e), hnorm(h))) (input columns sharded -> AR), then one SWA
    transformer block (TPLayer) and final_layernorm; logits via the shared (vocab-sharded) lm_head."""

    def __init__(self, j, bb, rank, W, sc, ropes):
        ref = _MTPRef(j, bb, ref_dev := torch.device("cuda", torch.cuda.current_device()))
        self.layer = TPLayer(ref, rank, W, sc, ropes)
        self.enorm, self.hnorm, self.fnorm = cp(ref.enorm), cp(ref.hnorm), cp(ref.fnorm)
        k = 2 * H // W
        self.k0 = rank * k
        self.eh = cp(ref.eh[:, self.k0:self.k0 + k])
        del ref
        torch.cuda.empty_cache()
        f = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=ref_dev)
        self.cat = f(2 * H, dt=torch.bfloat16)
        self.X = f(1, H, dt=torch.bfloat16); self.D = f(H); self.A = f(H)
        self.hout = f(1, H, dt=torch.bfloat16)


class TPModel:
    """Per-rank model. All ranks run the identical host loop (same inputs -> same tokens)."""

    def __init__(self, rank, W, layers=None, threads=16):
        self.rank, self.W = rank, W
        self.dev = dev = torch.device("cuda", torch.cuda.current_device())
        nl = layers or NL
        bb = M.Backbone()
        self.embed = bb.get("model.embed_tokens.weight", dev)
        self.norm = bb.get("model.norm.weight", dev)
        lm = bb.get("lm_head.weight", "cpu")
        V = lm.shape[0]; vr = (V + W - 1) // W
        self.v0 = rank * vr
        self.lm_head = lm[self.v0:self.v0 + vr].to(dev).contiguous(); del lm
        self.sc = TPScratch(dev, W)
        gu, dn = (int(v) for v in os.environ.get("MIMO_TP_WIDE", "0,1").split(","))
        MX().set_wide(gu, dn)
        kg, kd = (int(v) for v in os.environ.get("MIMO_TP_KSPLIT", "1,1").split(","))
        MX().set_ksplit(kg, kd)
        self.ropes = {False: RopeTable(CFG["rope_theta"], dev, LMAX), True: RopeTable(CFG["swa_rope_theta"], dev, LMAX)}
        self.layers = []
        pool = ThreadPoolExecutor(threads)
        t0 = time.time()
        for l in range(nl):
            ref = M.Layer(l, bb, dev, pool)
            self.layers.append(TPLayer(ref, rank, W, self.sc, self.ropes))
            del ref
            torch.cuda.empty_cache()
            if W > 1 and l % 5 == 4:
                dist.barrier()  # keep ranks in lockstep so the page cache serves the other ranks
            if rank == 0 and (l % 10 == 0 or l == nl - 1):
                log(f"[tp{W}] loaded layer {l} ({time.time()-t0:.0f}s) mem {torch.cuda.memory_allocated()/2**30:.1f} GiB")
        f = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)
        self.X = f(1, H, dt=torch.bfloat16); self.D = f(H); self.A = f(H)
        self.hf = f(1, H, dt=torch.bfloat16)
        self.lg = f(self.lm_head.shape[0])
        self.cand = f(2); self.cands = f(2 * W)
        self.st = {"pos": f(1, dt=torch.long), "slot_swa": f(1, dt=torch.long),
                   "n_full": f(1, dt=torch.int32), "n_swa": f(1, dt=torch.int32)}
        self.tok_in = f(1, dt=torch.long); self.tok_out = f(1, dt=torch.long)
        self.graph = None
        self.nvocab = V
        self.sampling = os.environ.get("MIMO_SAMPLING", "0") == "1"
        self.inv_t = f(1) + 1.0; self.noise_on = f(1)
        self.gen = None
        if self.sampling:
            self.gen = torch.Generator(device=dev); self.gen.manual_seed(1234 + 7919 * rank)
        self.fed = []  # token ids whose KV is resident at positions 0..len-1 (prefix reuse)
        self.car = None
        if W > 1 and os.environ.get("MIMO_CAR", "1") == "1":
            import car
            car.setup(rank, W, H * int(os.environ.get("MIMO_CAR_ROWS", "8")))
            self.car = car
            self.candp = f(32)  # all-gather-by-sum buffer (car needs n % 4 == 0)
        self.mtp = None
        if os.environ.get("MIMO_MTP", "0") == "1":
            self.mtp = TPMTP(0, bb, rank, W, self.sc, self.ropes)
            self.mst = {k: v.clone() for k, v in self.st.items()}
            self.mtok = f(1, dt=torch.long)
            if rank == 0:
                log(f"[tp{W}] MTP layer loaded, mem {torch.cuda.memory_allocated()/2**30:.1f} GiB")

    def ar(self, t, pft=(), pfb=()):
        if self.W > 1:
            if self.car is not None:
                self.car.allreduce(t, pft=pft, pfb=pfb)
            else:
                dist.all_reduce(t)

    def _argmax_tp(self, h, out_tok):
        """out_tok = global argmax of lm_head(h) over the vocab-sharded head (greedy)."""
        Kn.bf16_gemv(h.view(-1), self.lm_head, self.lg, round_out=True)
        lg = self.lg[:self.nvocab - self.v0] if self.v0 + self.lg.numel() > self.nvocab else self.lg
        m, i = torch.max(lg, 0)
        self.cand[0] = m; self.cand[1] = (i + self.v0).float()
        if self.car is not None:
            self.candp.zero_()
            self.candp[2 * self.rank:2 * self.rank + 2].copy_(self.cand)
            self.car.allreduce(self.candp, nb=1)
            self.cands.copy_(self.candp[:2 * self.W])
        elif self.W > 1:
            dist.all_gather_into_tensor(self.cands, self.cand)
        else:
            self.cands.copy_(self.cand)
        c = self.cands.view(self.W, 2)
        j = torch.argmax(c[:, 0])
        out_tok.copy_(c[:, 1].index_select(0, j.view(1)).long())

    def mtp_step(self, tok, hid, pos, logits=True):
        """Eager MTP step at position pos: input token tok (long [1]) + hidden hid (bf16 [1,H], post-final-norm
        hidden of the target (or previous MTP output)). Returns (draft token tensor [1] or None, mtp hidden [1,H])."""
        mt, st = self.mtp, self.mst
        st["pos"].fill_(pos); st["slot_swa"].fill_(pos % SWA)
        st["n_full"].fill_(pos + 1); st["n_swa"].fill_(min(pos + 1, SWA))
        e = self.embed[tok].view(1, H)
        # rmsnorm each half into the cat buffer (no residual)
        Kn.add_rmsnorm(e, mt.enorm, EPS, h=mt.cat[:H].view(1, H))
        Kn.add_rmsnorm(hid.view(1, H), mt.hnorm, EPS, h=mt.cat[H:].view(1, H))
        Kn.bf16_gemv(mt.cat[mt.k0:mt.k0 + mt.eh.shape[1]], mt.eh, mt.D)
        self.ar(mt.D)
        mt.X.zero_()
        L = mt.layer
        L.attn_part(mt.X, mt.D, st, mt.A)  # X = bf16(eh sum); h = input_layernorm(X)
        self.ar(mt.A)
        L.mlp_part(mt.X, mt.A, mt.D)
        self.ar(mt.D)
        Kn.add_rmsnorm(mt.X, mt.fnorm, EPS, d=mt.D, xo=mt.X, h=mt.hout)
        if logits:
            self._argmax_tp(mt.hout, self.mtok)
            return self.mtok, mt.hout
        return None, mt.hout

    def _step(self):
        st = self.st; p = st["pos"]
        torch.remainder(p, SWA, out=st["slot_swa"])
        st["n_full"].copy_(p + 1); st["n_swa"].copy_(torch.clamp(p + 1, max=SWA))
        self.X.copy_(self.embed[self.tok_in].view(1, H))
        self.D.zero_()
        MB = 2 ** 20
        nl = len(self.layers)

        def pfl(which, li):
            r = []
            for kind, mb in PF[which]:
                Ln = self.layers[li + 1] if li + 1 < nl else None
                t = {"q": Ln.qkv_w if Ln else None, "o": Ln.o_w if Ln else None,
                     "g": (self.layers[li] if which == "A" else Ln)}[kind]
                if kind == "g":
                    t = t.gate_w if (t is not None and t.moe) else None
                if t is not None and mb > 0:
                    r.append((t, mb * MB))
            return [t for t, _ in r], [b for _, b in r]
        fuse = FUSENORM and self.car is not None
        for li, L in enumerate(self.layers):
            L.attn_part(self.X, self.D, self.st, self.A, normed=fuse and li > 0)
            if fuse:
                self.car.allreduce(self.A, *pfl("A", li), norm=(self.X, L.ln2, L.sc.h, EPS))
            else:
                self.ar(self.A, *pfl("A", li))
            L.mlp_part(self.X, self.A, self.D, normed=fuse)
            if fuse:
                nxt = (self.layers[li + 1].ln1, L.sc.h) if li + 1 < nl else (self.norm, self.hf)
                self.car.allreduce(self.D, *pfl("D", li), norm=(self.X, nxt[0], nxt[1], EPS))
            else:
                self.ar(self.D, *pfl("D", li))
        if fuse:
            h = self.hf
        else:
            _, h = Kn.add_rmsnorm(self.X, self.norm, EPS, d=self.D, xo=self.X, h=self.hf)
        Kn.bf16_gemv(h.view(-1), self.lm_head, self.lg, round_out=True)
        lg = self.lg[:self.nvocab - self.v0] if self.v0 + self.lg.numel() > self.nvocab else self.lg
        if self.sampling:
            # Gumbel-max over the (vocab-sharded) logits, independent noise per rank; noise_on=0 -> exact greedy
            u = torch.rand(lg.shape, device=lg.device, generator=self.gen).clamp_(1e-10, 1.0 - 1e-7)
            lg = lg * self.inv_t - torch.log(-torch.log(u)) * self.noise_on
        m, i = torch.max(lg, 0)
        self.cand[0] = m; self.cand[1] = (i + self.v0).float()
        if self.car is not None:
            self.candp.zero_()
            self.candp[2 * self.rank:2 * self.rank + 2].copy_(self.cand)
            self.car.allreduce(self.candp, nb=1)
            self.cands.copy_(self.candp[:2 * self.W])
        elif self.W > 1:
            dist.all_gather_into_tensor(self.cands, self.cand)
        else:
            self.cands.copy_(self.cand)
        c = self.cands.view(self.W, 2)
        j = torch.argmax(c[:, 0])
        self.tok_out.copy_(c[:, 1].index_select(0, j.view(1)).long())
        # self-feeding: next replay decodes tok_out at pos+1 unless the host overrides tok_in
        p.add_(1)
        self.tok_in.copy_(self.tok_out)

    def capture(self):
        self.st["pos"].zero_(); self.tok_in.zero_()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        if os.environ.get("MIMO_GRAPHS", "1") != "1":
            return
        g = torch.cuda.CUDAGraph()
        if self.gen is not None:
            g.register_generator_state(self.gen)
        with torch.cuda.graph(g):
            self._step()
        torch.cuda.synchronize()
        self.graph = g
        self.reset()

    def reset(self):
        for L in self.layers:
            L.kc.zero_(); L.vc.zero_()

    def replay(self):
        if self.graph is not None:
            self.graph.replay()
        else:
            self._step()

    def set_sampling(self, temperature, seed=None):
        if temperature and temperature > 0:
            assert self.sampling, "start with MIMO_SAMPLING=1 for temperature > 0"
            self.inv_t.fill_(1.0 / temperature); self.noise_on.fill_(1.0)
            if seed is not None:
                self.gen.manual_seed(int(seed) * 4099 + 1234 + 7919 * self.rank)
        else:
            self.inv_t.fill_(1.0); self.noise_on.fill_(0.0)

    @torch.inference_mode()
    def generate(self, ids, max_new, stop=(151643, 151645, 151672), return_times=False, on_token=None,
                 reuse_prefix=False, temperature=0.0, seed=None):
        """Greedy (or Gumbel-max sampling when MIMO_SAMPLING=1 and temperature>0). The prompt is consumed
        token-by-token through the (self-feeding) decode graph. on_token(tok, i) -> False stops early; it must be
        deterministic across ranks (same tokens on every rank). reuse_prefix: if ids extends the previously fed
        sequence exactly, skip re-feeding it (SWA ring + full KV are then valid)."""
        if len(ids) + max_new > LMAX:
            raise ValueError("context too long")
        self.set_sampling(temperature, seed)
        k = 0
        if reuse_prefix and self.fed and len(ids) > len(self.fed) and ids[:len(self.fed)] == self.fed:
            k = len(self.fed)
        if k == 0:
            self.reset()
        self.cached = k
        self.st["pos"].fill_(k)
        pd = torch.tensor(ids, dtype=torch.long, device=self.dev)
        t0 = time.time()
        for i in range(k, len(ids)):
            self.tok_in.copy_(pd[i:i + 1])
            self.replay()
        nxt = self.tok_out.item()
        fed = list(ids)
        t_pre = time.time() - t0
        out, times = [], []
        for _ in range(max_new):
            out.append(nxt)
            if nxt in stop or len(out) >= max_new:
                break
            if on_token is not None and on_token(nxt, len(out) - 1) is False:
                break
            ts = time.perf_counter()
            self.replay()
            fed.append(nxt)
            nxt = self.tok_out.item()
            times.append(time.perf_counter() - ts)
        else:
            pass
        if on_token is not None and out and (out[-1] in stop or len(out) >= max_new):
            on_token(out[-1], len(out) - 1)
        self.fed = fed
        if return_times:
            return out, t_pre, times
        return out


def init_dist():
    rank = int(os.environ.get("RANK", 0)); W = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    if W > 1:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    return rank, W

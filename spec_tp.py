"""MTP speculative decoding for the TP4 MiMo model (greedy, lossless w.r.t. the verifier's own argmax).

One self-feeding CUDA graph per step processes MM = K+1 rows: [x0 = last emitted token at p0, d1..dK drafts]
through all 70 layers with M-row kernels (kernels_m.py, moe_ext.moe_cold_m), takes the per-row argmax g,
accepts n = #leading drafts with d_{i+1} == g_i (or a host-forced n during chunked prefill), then runs the MTP
layer batched over the MM rows (input (emb(next token of row i), hidden_i) at position p0+i) and a K-1 step
single-row draft chain from row n. Host reads [n, g0..gK] once per step and emits g[0..n].

SWA KV (main and MTP) uses a ring of R = SWA + MAXM slots (slot = pos % R) with positional masking, so rejected
drafts / padded rows (always at positions <= pmax that the next step rewrites) never clobber in-window keys.
Also provides an MM=1 graph (K=0) on the same kernels + KV layout for temperature sampling.
"""
import os, time
import torch
import torch.nn.functional as F

import mimo_tp as TP
import kernels as Kn
import kernels_m as KM
from mimo_tp import H, NH, NKV, HD, VD, ROPE, NE, TOPK, EPS, SWA, VSCALE, ROWS_G, SB_G, IM, MX

MAXM = 8
SPEC_NB = int(os.environ.get("MIMO_SPEC_CAR_NB", "1"))
R = SWA + MAXM


def fp8_block_quant(w, chunk=4096):
    """[N,K] -> (fp8 [N,K], fp32 scale [ceil(N/128), K/128]) with 128x128 blocks; for fp8_gemv_md(rows_g=N, sb_g=ceil(N/128))."""
    N, K = w.shape
    assert K % 128 == 0
    nb = (N + 127) // 128
    q = torch.empty(N, K, dtype=torch.float8_e4m3fn, device=w.device)
    sc = torch.empty(nb, K // 128, dtype=torch.float32, device=w.device)
    for r0 in range(0, N, chunk):
        r1 = min(N, r0 + chunk)
        b = w[r0:r1].float()
        pad = (-b.shape[0]) % 128
        if pad:
            b = torch.cat((b, b.new_zeros(pad, K)))
        bb = b.view(-1, 128, K // 128, 128)
        s_ = bb.abs().amax(dim=(1, 3)).clamp_min(1e-12) / 448.0
        qq = (bb / s_[:, None, :, None]).clamp_(-448, 448).view(-1, K)[:r1 - r0]
        q[r0:r1] = qq.to(torch.float8_e4m3fn)
        sc[r0 // 128:r0 // 128 + s_.shape[0]] = s_
        del b, bb, qq
    return q, sc


class SpecScratch:
    def __init__(self, dev, W, MM):
        f = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)
        nkv, nh, ir = NKV // W, NH // W, IM // W
        S = MM * TOPK
        self.MM, self.S, self.ir = MM, S, ir
        self.h = f(MM, H, dt=torch.bfloat16)
        self.qkv = f(MM, nkv * ROWS_G, dt=torch.bfloat16)
        self.q = f(MM, nh, HD, dt=torch.bfloat16)
        self.ao = f(MM, nh * VD, dt=torch.bfloat16)
        self.attn = KM.AttnM(dev, MM, nkv)
        self.logits = f(MM, NE); self.sel = f(S, dt=torch.long); self.wt = f(S)
        self.Hgu = f(S, 2 * ir); self.Hd = f(S, H)
        self.runs = f(2 + 2 * S + 1, dt=torch.int32); self.xr = f(S * 2 * H, dt=torch.half)
        self.gu_raw = f(S * 2 * ir); self.act = f(S * 3 * ir, dt=torch.half); self.d_raw = f(S * H)


def rows(sc, m):
    """View of the scratch for the first m rows."""
    class V: pass
    v = V()
    v.h, v.qkv, v.q, v.ao = sc.h[:m], sc.qkv[:m], sc.q[:m], sc.ao[:m]
    v.attn = sc.attn; v.logits = sc.logits[:m]; v.sel = sc.sel[:m * TOPK]; v.wt = sc.wt[:m * TOPK]
    v.Hgu, v.Hd = sc.Hgu[:m * TOPK], sc.Hd[:m * TOPK]
    v.runs, v.xr, v.gu_raw, v.act, v.d_raw = sc.runs, sc.xr, sc.gu_raw, sc.act, sc.d_raw
    v.ir = sc.ir
    return v


def attn_m(L, sc, X, D, st, A):
    KM.add_rmsnorm_m(X, L.ln1, EPS, d=D, xo=X, h=sc.h)
    KM.fp8_gemv_md(sc.h, L.qkv_w, L.qkv_s, ROWS_G, SB_G, sc.qkv)
    slot = st["slot_swa"] if L.swa else st["pos"]
    KM.qkv_post_m(sc.qkv, L.rope.cos, L.rope.sin, st["pos"], slot, sc.q, L.kc, L.vc, VSCALE, L.nkv, ROWS_G,
                  NH // NKV, HD, VD, ROPE)
    sc.attn(sc.q, L.kc, L.vc, st["pos"], st["pmax"], L.swa, R, SWA, L.sink, sc.ao, HD ** -0.5)
    KM.bf16_gemv_md(sc.ao, L.o_w, A)


def mlp_m(L, sc, X, A, D):
    KM.add_rmsnorm_m(X, L.ln2, EPS, d=A, xo=X, h=sc.h)
    h = sc.h
    if not L.moe:
        gu = torch.mm(h, L.gu_w.T)
        ir = L.gu_w.shape[0] // 2
        a = (F.silu(gu[:, :ir]) * gu[:, ir:]).contiguous()
        KM.bf16_gemv_md(a, L.d_w, D)
        return
    ir = sc.ir
    KM.gate_gemv_m(h, L.gate_w, sc.logits)
    KM.route_m(sc.logits, L.gate_b, sc.sel, sc.wt, TOPK)
    hot_y = None
    if L.has_hot:
        t = L.ht
        g, b = TP.HOT_GU, TP.HOT_DN
        Kn.nvfp4_gemv(h, L.hot_slot, sc.sel, t["w13_packed"], t["w13_bscale"], t["w13_scale2"], 0, 0, ir, sc.Hgu[:, :ir],
                      BN=g[0], BK=g[1], num_warps=g[2], xdiv=TOPK)
        Kn.nvfp4_gemv(h, L.hot_slot, sc.sel, t["w13_packed"], t["w13_bscale"], t["w13_scale2"], 1, ir, ir, sc.Hgu[:, ir:],
                      BN=g[0], BK=g[1], num_warps=g[2], xdiv=TOPK)
        Kn.nvfp4_gemv(sc.Hgu, L.hot_slot, sc.sel, t["w2_packed"], t["w2_bscale"], t["w2_scale2"], 0, 0, H, sc.Hd,
                      act_off=ir, BN=b[0], BK=b[1], num_warps=b[2])
        hot_y = sc.Hd
    MX().moe_cold_m(h, sc.sel, sc.wt, TOPK, L.gu_ptr, L.gu_k2, L.dn_ptr, L.dn_meta, L.maxn, sc.runs, sc.xr, sc.gu_raw,
                    sc.act, sc.d_raw, hot_y, L.is_hot, None, D, 65000.0, ir)


class Spec:
    def __init__(self, model, K):
        self.m = m = model
        self.K = K
        self.W, self.rank, self.dev = m.W, m.rank, m.dev
        dev = self.dev
        assert K + 1 <= MAXM
        assert K == 0 or m.mtp is not None, "MTP layer not loaded (MIMO_MTP=1)"
        f = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)
        MM = K + 1
        self.MM = MM
        # SWA rings of R slots for every SWA layer (main + MTP)
        for L in m.layers + ([m.mtp.layer] if m.mtp is not None else []):
            if L.swa and L.kc.shape[1] != R:
                L.kc = torch.zeros(L.kc.shape[0], R, HD, dtype=torch.bfloat16, device=dev)
                L.vc = torch.zeros(L.vc.shape[0], R, VD, dtype=torch.bfloat16, device=dev)
        torch.cuda.empty_cache()
        self.sc = SpecScratch(dev, self.W, MAXM)
        self.X = f(MAXM, H, dt=torch.bfloat16); self.D = f(MAXM, H); self.A = f(MAXM, H)
        self.hf = f(MAXM, H, dt=torch.bfloat16)
        self.lg = f(MAXM, m.lm_head.shape[0])
        self.nv = m.nvocab - m.v0 if m.v0 + m.lm_head.shape[0] > m.nvocab else m.lm_head.shape[0]
        np4 = (2 * MAXM * self.W + 3) // 4 * 4
        self.candp = f(np4)
        # device state
        self.tok = f(MAXM, dt=torch.long)          # rows' input tokens
        self.p0 = f(1, dt=torch.long)
        self.ar_i = torch.arange(MAXM, device=dev)
        self.pf_on = f(1, dt=torch.long); self.force_n = f(1, dt=torch.long)
        self.pf_next = f(1, dt=torch.long) - 1     # prefill: first token of the next chunk (-1: none)
        self.out = f(1 + MAXM, dt=torch.long)      # [n, g0..]
        self.st = {"pos": f(MAXM, dt=torch.long), "slot_swa": f(MAXM, dt=torch.long), "pmax": f(1, dt=torch.long)}
        self.cst = {"pos": f(1, dt=torch.long), "slot_swa": f(1, dt=torch.long), "pmax": f(1, dt=torch.long)}
        if m.mtp is not None:
            self.mX = f(MAXM, H, dt=torch.bfloat16); self.mD = f(MAXM, H); self.mA = f(MAXM, H)
            self.ce = f(MAXM, H, dt=torch.bfloat16); self.chh = f(MAXM, H, dt=torch.bfloat16)
            self.hm = f(MAXM, H, dt=torch.bfloat16)
            self.dtok = f(1, dt=torch.long)
        # drafts only change acceptance, never the emitted tokens -> fp8 draft lm_head + fp8 MTP dense MLP
        self.draft_fp8 = m.mtp is not None and os.environ.get("MIMO_DRAFT_FP8", "1") == "1"
        if self.draft_fp8:
            self.lm8, self.lm8s = fp8_block_quant(m.lm_head)
            L = m.mtp.layer
            L.gu8, L.gu8s = fp8_block_quant(L.gu_w)
            L.d8, L.d8s = fp8_block_quant(L.d_w)
            self.mgu = f(MAXM, L.gu_w.shape[0], dt=torch.bfloat16)
            torch.cuda.empty_cache()
        self.sampling = m.sampling
        self.graphs = {}
        self.PF = int(os.environ.get("MIMO_SPEC_PF", str(MAXM))) if K > 0 else 0  # prefill graph rows
        assert self.PF <= MAXM
        self.fed = []; self.hw = -1; self.cached = 0
        self.pre_step = None  # test hook: f(spec, out_so_far) before each decode replay

    def ar(self, t, pft=(), pfb=()):
        # multi-row all-reduce: MIMO_SPEC_CAR_NB blocks when > 1 row (needs MIMO_CAR_NB >= it at car.setup)
        if self.W > 1 and self.m.car is not None and t.dim() == 2 and t.shape[0] > 1:
            self.m.car.allreduce(t, nb=SPEC_NB, pft=pft, pfb=pfb)
        else:
            self.m.ar(t, pft, pfb)

    def _pfl(self, which, li):
        """L2 prefetch ranges issued by extra blocks of the all-reduce (same policy as TPModel, MIMO_PF)."""
        ls, r = self.m.layers, []
        Ln = ls[li + 1] if li + 1 < len(ls) else None
        for kind, mb in TP.PF[which]:
            t = {"q": Ln.qkv_w if Ln else None, "o": Ln.o_w if Ln else None, "g": (ls[li] if which == "A" else Ln)}[kind]
            if kind == "g":
                t = t.gate_w if (t is not None and t.moe) else None
            if t is not None and mb > 0:
                r.append((t, mb * 2 ** 20))
        return [t for t, _ in r], [b for _, b in r]

    # ------------------------------------------------------------------ pieces
    def _argmax_rows(self, h, mm, sample=False, draft=False):
        """[mm] long global argmax of lm_head(h[:mm]) (vocab-sharded)."""
        m = self.m
        if draft and self.draft_fp8:
            N = self.lm8.shape[0]
            lg = KM.fp8_gemv_md(h[:mm], self.lm8, self.lm8s, N, (N + 127) // 128, self.lg[:mm], BN=32)[:, :self.nv]
        else:
            lg = KM.bf16_gemv_md(h[:mm], m.lm_head, self.lg[:mm], round_out=True)[:, :self.nv]
        if sample:
            u = torch.rand(lg.shape, device=lg.device, generator=m.gen).clamp_(1e-10, 1.0 - 1e-7)
            lg = lg * m.inv_t - torch.log(-torch.log(u)) * m.noise_on
        mx, ix = torch.max(lg, 1)
        W = self.W
        if W > 1:
            self.candp.zero_()
            c = self.candp[:2 * mm * W].view(W, mm, 2)
            c[self.rank, :, 0] = mx
            c[self.rank, :, 1] = (ix + m.v0).float()
            m.car.allreduce(self.candp, nb=1)
            j = torch.argmax(c[:, :, 0], 0)  # [mm]
            return c[:, :, 1].gather(0, j.view(1, mm)).view(mm).long()
        return ix + m.v0

    def _mtp_rows(self, toks, hid, st, mm):
        """MTP layer on mm rows: hm[:mm] = final_layernorm(block(eh_proj(cat(enorm(emb), hnorm(hid)))))"""
        m, mt = self.m, self.m.mtp
        sc = rows(self.sc, mm)
        e = m.embed.index_select(0, toks).view(mm, H)
        KM.add_rmsnorm_m(e, mt.enorm, EPS, h=self.ce[:mm])
        KM.add_rmsnorm_m(hid, mt.hnorm, EPS, h=self.chh[:mm])
        kw = mt.eh.shape[1]
        if mt.k0 + kw <= H:
            xin = self.ce[:mm, mt.k0:mt.k0 + kw]
        elif mt.k0 >= H:
            xin = self.chh[:mm, mt.k0 - H:mt.k0 - H + kw]
        else:  # W=1 test path
            xin = torch.cat((self.ce[:mm], self.chh[:mm]), 1)[:, mt.k0:mt.k0 + kw]
        KM.bf16_gemv_md(xin, mt.eh, self.mD[:mm])
        self.ar(self.mD[:mm])
        X = self.mX[:mm]; X.zero_()
        attn_m(mt.layer, sc, X, self.mD[:mm], st, self.mA[:mm])
        self.ar(self.mA[:mm])
        if self.draft_fp8:
            L = mt.layer
            KM.add_rmsnorm_m(X, L.ln2, EPS, d=self.mA[:mm], xo=X, h=sc.h)
            N = L.gu8.shape[0]; ir = N // 2
            gu = KM.fp8_gemv_md(sc.h, L.gu8, L.gu8s, N, (N + 127) // 128, self.mgu[:mm], BN=32)
            a = (F.silu(gu[:, :ir]) * gu[:, ir:]).contiguous()
            N2 = L.d8.shape[0]
            KM.fp8_gemv_md(a, L.d8, L.d8s, N2, (N2 + 127) // 128, self.mD[:mm], BN=32)
        else:
            mlp_m(mt.layer, sc, X, self.mA[:mm], self.mD[:mm])
        self.ar(self.mD[:mm])
        KM.add_rmsnorm_m(X, mt.fnorm, EPS, d=self.mD[:mm], xo=X, h=self.hm[:mm])
        return self.hm[:mm]

    def _step(self, MM, sample=False, chain=True):
        m, st = self.m, self.st
        K = MM - 1
        sc = rows(self.sc, MM)
        pos = st["pos"][:MM]
        torch.add(self.p0, self.ar_i[:MM], out=pos)
        torch.remainder(pos, R, out=st["slot_swa"][:MM])
        torch.add(self.p0, K, out=st["pmax"])
        sst = {"pos": pos, "slot_swa": st["slot_swa"][:MM], "pmax": st["pmax"]}
        tok = self.tok[:MM]
        X, D, A = self.X[:MM], self.D[:MM], self.A[:MM]
        torch.index_select(m.embed, 0, tok, out=X)
        D.zero_()
        for li, L in enumerate(m.layers):
            attn_m(L, sc, X, D, sst, A)
            self.ar(A, *self._pfl("A", li))
            mlp_m(L, sc, X, A, D)
            self.ar(D, *self._pfl("D", li))
        hf = self.hf[:MM]
        KM.add_rmsnorm_m(X, m.norm, EPS, d=D, xo=X, h=hf)
        g = self._argmax_rows(hf, MM, sample)
        if K > 0:
            match = (tok[1:] == g[:-1]).long()
            nver = torch.cumprod(match, 0).sum().view(1)
        else:
            nver = torch.zeros(1, dtype=torch.long, device=self.dev)
        n = torch.where(self.pf_on > 0, self.force_n, nver)
        self.out[0:1].copy_(n)
        self.out[1:1 + MM].copy_(g)
        gn = g.index_select(0, n)
        if K > 0:
            ar = self.ar_i[:MM]
            tsh = torch.cat((tok[1:], tok[-1:]))
            gsub = torch.where(self.pf_next >= 0, self.pf_next, g)
            nt = torch.where(ar < n, tsh, gsub)
            hm = self._mtp_rows(nt, hf, sst, MM)
            h1 = hm.index_select(0, n)
            d = self._argmax_rows(h1, 1, draft=True)
            self.tok[1:2].copy_(d)
            cst = self.cst
            for j in range(1, K if chain else 1):
                cp_ = self.p0 + n + j
                cst["pos"].copy_(cp_); torch.remainder(cp_, R, out=cst["slot_swa"]); cst["pmax"].copy_(cp_)
                h1 = self._mtp_rows(self.tok[j:j + 1], h1.clone(), cst, 1)
                d = self._argmax_rows(h1, 1, draft=True)
                self.tok[j + 1:j + 2].copy_(d)
        self.tok[0:1].copy_(gn)
        self.p0.add_(n + 1)

    def capture(self, MM, pf=False):
        """pf=True: prefill-only graph (forced acceptance, no K-1 draft chain), stored under key -MM."""
        m = self.m
        self.p0.zero_(); self.tok.zero_(); self.pf_on.fill_(1); self.force_n.fill_(MM - 1)
        sample = MM == 1 and self.sampling
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step(MM, sample, chain=not pf)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        if sample:
            g.register_generator_state(m.gen)
        with torch.cuda.graph(g):
            self._step(MM, sample, chain=not pf)
        torch.cuda.synchronize()
        self.graphs[-MM if pf else MM] = g
        self.reset()

    def capture_all(self):
        self.capture(self.MM)
        if self.MM != 1:
            self.capture(1)
        if self.PF > self.MM:
            self.capture(self.PF, pf=True)

    def reset(self):
        m = self.m
        for L in m.layers + ([m.mtp.layer] if m.mtp is not None else []):
            L.kc.zero_(); L.vc.zero_()
        self.fed = []; self.hw = -1

    def replay(self, MM):
        self.graphs[MM].replay()
        self.hw = max(self.hw, int(self._p0h) + abs(MM) - 1)

    # ------------------------------------------------------------------ host loop
    @torch.inference_mode()
    def generate(self, ids, max_new, stop=(151643, 151645, 151672), return_times=False, on_token=None,
                 reuse_prefix=False, temperature=0.0, seed=None):
        m = self.m
        if len(ids) + max_new + MAXM > TP.LMAX:
            raise ValueError("context too long")
        m.set_sampling(temperature, seed)
        MM = 1 if (temperature and temperature > 0) else self.MM
        k = 0
        if (reuse_prefix and self.fed and len(ids) > len(self.fed) and ids[:len(self.fed)] == self.fed
                and len(self.fed) + MM - 1 >= self.hw):
            k = len(self.fed)
        if k == 0:
            self.reset()
        self.cached = k
        m.cached = k
        t0 = time.time()
        # chunked prefill of ids[k:] through the MM graph (forced acceptance)
        self.pf_on.fill_(1)
        i = k
        out = []
        pfg = -self.PF if (-self.PF) in self.graphs else 0
        while i < len(ids):
            r = len(ids) - i
            if pfg and r > MM:  # wide prefill-only graph; the last <= MM tokens go through the decode graph
                c, G = min(self.PF, r - MM), pfg
            else:
                c, G = min(MM, r), MM
            self.tok[:c].copy_(torch.tensor(ids[i:i + c], dtype=torch.long), non_blocking=False)
            self.p0.fill_(i); self._p0h = i
            self.force_n.fill_(c - 1)
            self.pf_next.fill_(ids[i + c] if i + c < len(ids) else -1)
            self.replay(G)
            i += c
        o = self.out[:1 + MM].tolist()
        nxt = o[1 + o[0]]
        self._p0h = len(ids)
        fed = list(ids)
        t_pre = time.time() - t0
        self.pf_on.fill_(0)
        times, steps, acc = [], 0, []
        out = [nxt]
        final = nxt in stop or len(out) >= max_new
        done = final or (on_token is not None and on_token(nxt, 0) is False)
        while not done:
            if self.pre_step is not None:
                self.pre_step(self, out)
            ts = time.perf_counter()
            self.replay(MM)
            o = self.out[:1 + MM].tolist()
            n = o[0]
            times.append(time.perf_counter() - ts)
            steps += 1; acc.append(n)
            fed.append(out[-1])
            new = o[1:2 + n]
            self._p0h += n + 1
            for j, t in enumerate(new):
                out.append(t)
                if t in stop or len(out) >= max_new:
                    final = done = True
                    break
                if on_token is not None and on_token(t, len(out) - 1) is False:
                    done = True
                    break
                if j < len(new) - 1:
                    fed.append(t)
        if on_token is not None and final:
            on_token(out[-1], len(out) - 1)
        self.fed = fed
        self.last_accept = acc
        if return_times:
            return out, t_pre, times
        return out

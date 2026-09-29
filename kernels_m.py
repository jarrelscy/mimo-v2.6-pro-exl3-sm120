"""M-token (M = 1 + #drafts, small) variants of the decode kernels for MTP speculative verification.
Every row is computed independently (row-local reductions), rows are consecutive positions pos[0..M).
SWA layers use a ring of R >= 128 + M slots (slot = pos % R) and mask by reconstructed slot position, so writing
M new tokens (some of which may be rejected drafts) never clobbers a key still inside any query's window."""
import torch
import triton
import triton.language as tl
from kernels import _bf, _fp4_lut  # noqa: F401


def _mp(M):
    return triton.next_power_of_2(M)


# ----------------------------------------------------------------------------- FP8 block-scaled GEMV, M rows
@triton.jit
def _fp8_gemv_m_kernel(x_ptr, x_rs, w_ptr, s_ptr, y_ptr, y_rs, N, K, s_stride, M,
                       ROWS_G: tl.constexpr, SB_G: tl.constexpr, MP: tl.constexpr,
                       BN: tl.constexpr, BK: tl.constexpr, ROUND_BF16: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    nmask = rn < N
    rm = tl.arange(0, MP)
    mm = rm < M
    g = rn // ROWS_G
    srow = g * SB_G + (rn - g * ROWS_G) // 128
    NS: tl.constexpr = BK // 128
    acc = tl.zeros((MP, BN, BK), dtype=tl.float32)
    rk = tl.arange(0, BK)
    rs = tl.arange(0, NS)
    for k0 in range(0, K, BK):
        w = tl.load(w_ptr + rn[:, None].to(tl.int64) * K + (k0 + rk)[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
        s = tl.load(s_ptr + srow[:, None] * s_stride + (k0 // 128 + rs)[None, :], mask=nmask[:, None], other=0.0)
        w = tl.reshape(tl.reshape(w, (BN, NS, 128)) * s[:, :, None], (BN, BK))
        if ROUND_BF16:
            w = w.to(tl.bfloat16).to(tl.float32)
        x = tl.load(x_ptr + rm[:, None] * x_rs + (k0 + rk)[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
        acc += w[None, :, :] * x[:, None, :]
    y = tl.sum(acc, 2)
    tl.store(y_ptr + rm[:, None] * y_rs + rn[None, :], y.to(y_ptr.dtype.element_ty), mask=mm[:, None] & nmask[None, :])


def fp8_gemv_m(x, w, s, rows_g, sb_g, out, BN=8, BK=512, round_bf16=True, num_warps=8):
    """x [M,K] bf16 (row stride any), w [N,K] fp8, out [M,N]."""
    N, K = w.shape
    M = x.shape[0]
    _fp8_gemv_m_kernel[(triton.cdiv(N, BN),)](x, x.stride(0), w, s, out, out.stride(0), N, K, s.stride(0), M,
                                              rows_g, sb_g, _mp(M), BN, BK, round_bf16, num_warps=num_warps)
    return out


# ----------------------------------------------------------------------------- bf16 GEMV, M rows, fp32 out
@triton.jit
def _bf16_gemv_m_kernel(x_ptr, x_rs, w_ptr, y_ptr, y_rs, N, K, M, MP: tl.constexpr,
                        BN: tl.constexpr, BK: tl.constexpr, ROUND: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    nmask = rn < N
    rm = tl.arange(0, MP)
    mm = rm < M
    rk = tl.arange(0, BK)
    acc = tl.zeros((MP, BN, BK), dtype=tl.float32)
    for k0 in range(0, K, BK):
        w = tl.load(w_ptr + rn[:, None].to(tl.int64) * K + (k0 + rk)[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
        x = tl.load(x_ptr + rm[:, None] * x_rs + (k0 + rk)[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
        acc += w[None, :, :] * x[:, None, :]
    y = tl.sum(acc, 2)
    if ROUND:
        y = _bf(y)
    tl.store(y_ptr + rm[:, None] * y_rs + rn[None, :], y, mask=mm[:, None] & nmask[None, :])


def bf16_gemv_m(x, w, out, BN=8, BK=512, round_out=False, num_warps=8):
    """out [M,N] fp32 = x [M,K] bf16 @ w[N,K]^T (K % BK == 0; x row stride any)."""
    N, K = w.shape
    M = x.shape[0]
    assert K % BK == 0
    _bf16_gemv_m_kernel[(triton.cdiv(N, BN),)](x, x.stride(0), w, out, out.stride(0), N, K, M, _mp(M), BN, BK,
                                               round_out, num_warps=num_warps)
    return out


# ----------------------------------------------------------------------------- router
@triton.jit
def _gate_gemv_m_kernel(x_ptr, w_ptr, y_ptr, N, M, K: tl.constexpr, MP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rm = tl.arange(0, MP)
    mm = rm < M
    rk = tl.arange(0, BK)
    acc = tl.zeros((MP, BN, BK), dtype=tl.float32)
    for k0 in range(0, K, BK):
        w = tl.load(w_ptr + rn[:, None] * K + (k0 + rk)[None, :])
        x = tl.load(x_ptr + rm[:, None] * K + (k0 + rk)[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
        acc += w[None, :, :] * x[:, None, :]
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], tl.sum(acc, 2), mask=mm[:, None])


def gate_gemv_m(x, w, out, BN=2, BK=1024, num_warps=8):
    """out [M,N] fp32 = float(x [M,K] bf16, contiguous) @ w[N,K] fp32^T."""
    N, K = w.shape
    M = x.shape[0]
    _gate_gemv_m_kernel[(N // BN,)](x, w, out, N, M, K, _mp(M), BN, BK, num_warps=num_warps)
    return out


@triton.jit
def _route_m_kernel(lg_ptr, b_ptr, idx_ptr, wt_ptr, NE, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    r = tl.arange(0, BLOCK)
    m = r < NE
    s = tl.sigmoid(tl.load(lg_ptr + t * NE + r, mask=m, other=0.0))
    key = tl.where(m, s + tl.load(b_ptr + r, mask=m, other=0.0), -float("inf"))
    rk = tl.arange(0, TOPK)
    wv = tl.zeros((TOPK,), tl.float32)
    iv = tl.zeros((TOPK,), tl.int64)
    for k in tl.static_range(TOPK):
        mx = tl.max(key, 0)
        i = tl.min(tl.where(key == mx, r, BLOCK), 0)
        sv = tl.sum(tl.where(r == i, s, 0.0), 0)
        wv = tl.where(rk == k, sv, wv)
        iv = tl.where(rk == k, i.to(tl.int64), iv)
        key = tl.where(r == i, -float("inf"), key)
    tot = tl.sum(wv, 0)
    tl.store(idx_ptr + t * TOPK + rk, iv)
    tl.store(wt_ptr + t * TOPK + rk, wv / (tot + 1e-20))


def route_m(logits, bias, idx_out, wt_out, topk=8):
    """logits [M, NE] -> idx_out [M*topk] int64, wt_out [M*topk] fp32."""
    M, NE = logits.shape
    _route_m_kernel[(M,)](logits, bias, idx_out, wt_out, NE, topk, triton.next_power_of_2(NE), num_warps=4)


# ----------------------------------------------------------------------------- residual add + rmsnorm, M rows
@triton.jit
def _add_rmsnorm_m_kernel(x_ptr, d_ptr, w_ptr, xo_ptr, h_ptr, N, eps, HAS_D: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    r = tl.arange(0, BLOCK)
    m = r < N
    x = tl.load(x_ptr + t * N + r, mask=m, other=0.0).to(tl.float32)
    if HAS_D:
        d = tl.load(d_ptr + t * N + r, mask=m, other=0.0).to(tl.float32)
        d = _bf(d)
        x = _bf(x + d)
        tl.store(xo_ptr + t * N + r, x.to(tl.bfloat16), mask=m)
    ms = tl.sum(x * x, 0) / N
    y = (x * tl.math.rsqrt(ms + eps)).to(tl.bfloat16).to(tl.float32)
    w = tl.load(w_ptr + r, mask=m, other=0.0).to(tl.float32)
    tl.store(h_ptr + t * N + r, (w * y).to(tl.bfloat16), mask=m)


def add_rmsnorm_m(x, w, eps, d=None, xo=None, h=None):
    """x [M,N] bf16 contiguous (d [M,N] fp32/bf16); xo = bf16(x + bf16(d)); h = rmsnorm(xo) * w."""
    M, N = x.shape
    _add_rmsnorm_m_kernel[(M,)](x, d if d is not None else x, w, xo if xo is not None else x, h, N, eps,
                                d is not None, triton.next_power_of_2(N), num_warps=16)
    return h


# ----------------------------------------------------------------------------- rope + KV write, M tokens
@triton.jit
def _qkv_post_m_kernel(qkv_ptr, qkv_rs, cos_ptr, sin_ptr, pos_ptr, slot_ptr, q_ptr, q_rs, kc_ptr, vc_ptr,
                       kc_sh, kc_ss, vc_sh, vc_ss, vscale, ROWS_G: tl.constexpr, G: tl.constexpr, HD: tl.constexpr,
                       VD: tl.constexpr, ROPE: tl.constexpr):
    g = tl.program_id(0)
    t = tl.program_id(1)
    pos = tl.load(pos_ptr + t)
    slot = tl.load(slot_ptr + t)
    RH: tl.constexpr = ROPE // 2
    rr = tl.arange(0, ROPE)
    rot_src = tl.where(rr < RH, rr + RH, rr - RH)
    sign = tl.where(rr < RH, -1.0, 1.0)
    cos = tl.load(cos_ptr + pos * ROPE + rr).to(tl.float32)
    sin = tl.load(sin_ptr + pos * ROPE + rr).to(tl.float32)
    rh = tl.arange(0, 2 * G)
    hm = rh <= G
    base = qkv_ptr + t * qkv_rs + g * ROWS_G
    r = tl.load(base + rh[:, None] * HD + rr[None, :], mask=hm[:, None], other=0.0).to(tl.float32)
    rs = tl.load(base + rh[:, None] * HD + rot_src[None, :], mask=hm[:, None], other=0.0).to(tl.float32) * sign[None, :]
    o = _bf(_bf(r * cos[None, :]) + _bf(rs * sin[None, :]))
    NR: tl.constexpr = HD - ROPE
    rn = tl.arange(0, NR)
    rest = tl.load(base + rh[:, None] * HD + ROPE + rn[None, :], mask=hm[:, None], other=0.0)
    isq = rh < G
    qrow = g * G + rh
    qb = q_ptr + t * q_rs
    tl.store(qb + qrow[:, None] * HD + rr[None, :], o.to(tl.bfloat16), mask=isq[:, None])
    tl.store(qb + qrow[:, None] * HD + ROPE + rn[None, :], rest, mask=isq[:, None])
    isk = rh == G
    kb = kc_ptr + g * kc_sh + slot * kc_ss
    tl.store(kb + rh[:, None] * 0 + rr[None, :], o.to(tl.bfloat16), mask=isk[:, None])
    tl.store(kb + rh[:, None] * 0 + ROPE + rn[None, :], rest, mask=isk[:, None])
    rv = tl.arange(0, VD)
    v = tl.load(base + (G + 1) * HD + rv).to(tl.float32)
    v = (v * vscale).to(tl.bfloat16)
    tl.store(vc_ptr + g * vc_sh + slot * vc_ss + rv, v)


def qkv_post_m(qkv, cos, sin, pos, slot, q_out, kc, vc, vscale, nkv, rows_g, g, hd, vd, rope):
    """qkv [M, nkv*rows_g] bf16, pos/slot int64 [M], q_out [M, nkv*g, hd]."""
    M = qkv.shape[0]
    _qkv_post_m_kernel[(nkv, M)](qkv, qkv.stride(0), cos, sin, pos, slot, q_out, q_out.stride(0), kc, vc,
                                 kc.stride(0), kc.stride(1), vc.stride(0), vc.stride(1), vscale, rows_g, g, hd, vd,
                                 rope, num_warps=4)


# ----------------------------------------------------------------------------- attention, M queries
@triton.jit
def _attn_m_kernel(q_ptr, q_rs, k_ptr, v_ptr, pos_ptr, pmax_ptr, o_ptr, m_ptr, l_ptr,
                   k_sh, k_ss, v_sh, v_ss, scale, NKV, R, WIN,
                   G: tl.constexpr, BS: tl.constexpr, NSPLIT: tl.constexpr, SWA: tl.constexpr):
    """Full layers: query at position p sees slots [0, p]. SWA ring (R slots, slot = pos % R): slot s holds position
    ps = pmax - ((pmax - s) mod R); valid iff 0 <= ps <= p and p - ps < WIN. Split length is fixed by n = R (SWA)
    or n = p + 1 (full), exactly like a batch-1 call at the same position."""
    kv = tl.program_id(0)
    sp = tl.program_id(1)
    t = tl.program_id(2)
    p = tl.load(pos_ptr + t)
    if SWA:
        n = R
        pmax = tl.load(pmax_ptr)
    else:
        n = p + 1
        pmax = p
    SPLIT_LEN = tl.cdiv(tl.cdiv(n, NSPLIT), BS) * BS
    lo = sp * SPLIT_LEN
    hi = tl.minimum(lo + SPLIT_LEN, n)
    rg = tl.arange(0, G)
    ra = tl.arange(0, 128)
    rb = tl.arange(0, 64)
    qbase = q_ptr + t * q_rs
    qa = tl.load(qbase + (kv * G + rg)[:, None] * 192 + ra[None, :])
    qb = tl.load(qbase + (kv * G + rg)[:, None] * 192 + 128 + rb[None, :])
    m_i = tl.full((G,), -float("inf"), tl.float32)
    l_i = tl.zeros((G,), tl.float32)
    acc = tl.zeros((G, 128), tl.float32)
    rs = tl.arange(0, BS)
    kbase = k_ptr + kv * k_sh
    vbase = v_ptr + kv * v_sh
    for s0 in range(lo, hi, BS):
        sl = s0 + rs
        sm = sl < hi
        if SWA:
            dd = (pmax - sl) % R
            dd = tl.where(dd < 0, dd + R, dd)
            ps = pmax - dd
            sm = sm & (ps >= 0) & (ps <= p) & (p - ps < WIN)
        ka = tl.load(kbase + sl[:, None] * k_ss + ra[None, :], mask=sm[:, None], other=0.0)
        kb = tl.load(kbase + sl[:, None] * k_ss + 128 + rb[None, :], mask=sm[:, None], other=0.0)
        sc = tl.dot(qa, tl.trans(ka)) + tl.dot(qb, tl.trans(kb))
        sc = sc.to(tl.bfloat16).to(tl.float32) * scale
        sc = tl.where(sm[None, :], sc, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        pr = tl.exp(sc - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(pr, 1)
        v = tl.load(vbase + sl[:, None] * v_ss + ra[None, :], mask=sm[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(pr.to(tl.bfloat16), v)
        m_i = m_new
    base = ((t * NKV + kv) * NSPLIT + sp) * G
    tl.store(o_ptr + (base + rg)[:, None] * 128 + ra[None, :], acc)
    tl.store(m_ptr + base + rg, m_i)
    tl.store(l_ptr + base + rg, l_i)


@triton.jit
def _attn_combine_m_kernel(o_ptr, m_ptr, l_ptr, sink_ptr, out_ptr, out_rs, NKV, G: tl.constexpr, NSPLIT: tl.constexpr,
                           HAS_SINK: tl.constexpr):
    h = tl.program_id(0)
    t = tl.program_id(1)
    kv = h // G
    i = h % G
    rsp = tl.arange(0, NSPLIT)
    idx = ((t * NKV + kv) * NSPLIT + rsp) * G + i
    m = tl.load(m_ptr + idx)
    l = tl.load(l_ptr + idx)
    Mx = tl.max(m, 0)
    if HAS_SINK:
        sk = tl.load(sink_ptr + h).to(tl.float32)
        Mx = tl.maximum(Mx, sk)
    w = tl.where(m == -float("inf"), 0.0, tl.exp(m - Mx))
    L = tl.sum(l * w, 0)
    if HAS_SINK:
        L += tl.exp(sk - Mx)
    ra = tl.arange(0, 128)
    o = tl.load(o_ptr + idx[:, None] * 128 + ra[None, :])
    o = tl.sum(o * w[:, None], 0) / L
    tl.store(out_ptr + t * out_rs + h * 128 + ra, o.to(tl.bfloat16))


class AttnM:
    def __init__(self, dev, M, nkv, g=16, nsplit=16):
        self.M, self.nkv, self.g, self.nsplit = M, nkv, g, nsplit
        self.o = torch.empty(M * nkv * nsplit * g, 128, dtype=torch.float32, device=dev)
        self.m = torch.empty(M * nkv * nsplit * g, dtype=torch.float32, device=dev)
        self.l = torch.empty(M * nkv * nsplit * g, dtype=torch.float32, device=dev)

    def __call__(self, q, kc, vc, pos, pmax, swa, R, win, sink, out, scale, BS=32):
        """q [M, NH, 192] bf16; out [M, NH*128] bf16."""
        M = q.shape[0]
        ns = self.nsplit
        _attn_m_kernel[(self.nkv, ns, M)](q, q.stride(0), kc, vc, pos, pmax, self.o, self.m, self.l,
                                          kc.stride(0), kc.stride(1), vc.stride(0), vc.stride(1), scale,
                                          self.nkv, R, win, self.g, BS, ns, swa, num_warps=4)
        _attn_combine_m_kernel[(self.nkv * self.g, M)](self.o, self.m, self.l, sink if sink is not None else self.m,
                                                       out, out.stride(0), self.nkv, self.g, ns, sink is not None,
                                                       num_warps=1)
        return out


# ----------------------------------------------------------------------------- tensor-core (tl.dot) variants, M <= 16
@triton.jit
def _fp8_gemv_md_kernel(x_ptr, x_rs, w_ptr, s_ptr, y_ptr, y_rs, N, K, s_stride, M,
                        ROWS_G: tl.constexpr, SB_G: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    nmask = rn < N
    rm = tl.arange(0, 16)
    mm = rm < M
    g = rn // ROWS_G
    srow = g * SB_G + (rn - g * ROWS_G) // 128
    NS: tl.constexpr = BK // 128
    acc = tl.zeros((16, BN), dtype=tl.float32)
    rk = tl.arange(0, BK)
    rs = tl.arange(0, NS)
    for k0 in range(0, K, BK):
        w = tl.load(w_ptr + rn[:, None].to(tl.int64) * K + (k0 + rk)[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
        s = tl.load(s_ptr + srow[:, None] * s_stride + (k0 // 128 + rs)[None, :], mask=nmask[:, None], other=0.0)
        w = tl.reshape(tl.reshape(w, (BN, NS, 128)) * s[:, :, None], (BN, BK)).to(tl.bfloat16)
        x = tl.load(x_ptr + rm[:, None] * x_rs + (k0 + rk)[None, :], mask=mm[:, None], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(y_ptr + rm[:, None] * y_rs + rn[None, :], acc.to(y_ptr.dtype.element_ty), mask=mm[:, None] & nmask[None, :])


def fp8_gemv_md(x, w, s, rows_g, sb_g, out, BN=16, BK=256, num_warps=4, stages=3):
    N, K = w.shape
    M = x.shape[0]
    _fp8_gemv_md_kernel[(triton.cdiv(N, BN),)](x, x.stride(0), w, s, out, out.stride(0), N, K, s.stride(0), M,
                                               rows_g, sb_g, BN, BK, num_warps=num_warps, num_stages=stages)
    return out


@triton.jit
def _bf16_gemv_md_kernel(x_ptr, x_rs, w_ptr, y_ptr, y_rs, N, K, M, BN: tl.constexpr, BK: tl.constexpr,
                         ROUND: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    nmask = rn < N
    rm = tl.arange(0, 16)
    mm = rm < M
    rk = tl.arange(0, BK)
    acc = tl.zeros((16, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        w = tl.load(w_ptr + rn[:, None].to(tl.int64) * K + (k0 + rk)[None, :], mask=nmask[:, None], other=0.0)
        x = tl.load(x_ptr + rm[:, None] * x_rs + (k0 + rk)[None, :], mask=mm[:, None], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    if ROUND:
        acc = _bf(acc)
    tl.store(y_ptr + rm[:, None] * y_rs + rn[None, :], acc, mask=mm[:, None] & nmask[None, :])


def bf16_gemv_md(x, w, out, BN=16, BK=256, round_out=False, num_warps=4, stages=3):
    N, K = w.shape
    M = x.shape[0]
    _bf16_gemv_md_kernel[(triton.cdiv(N, BN),)](x, x.stride(0), w, out, out.stride(0), N, K, M, BN, BK, round_out,
                                                num_warps=num_warps, num_stages=stages)
    return out

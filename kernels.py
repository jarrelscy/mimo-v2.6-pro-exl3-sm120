"""Triton decode kernels for MiMo-V2.6-Pro-EXL3 fast path (batch-1 decode)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _bf(x):
    # fp32 -> bf16 (RNE) -> fp32 via integer ops: a plain .to(bf16).to(fp32) pair gets folded by the
    # compiler when it feeds an add, which breaks bit-matching with the reference's bf16 op-by-op math
    u = x.to(tl.uint32, bitcast=True)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.to(tl.float32, bitcast=True)


# ----------------------------------------------------------------------------- FP8 block-scaled GEMV
@triton.jit
def _fp8_gemv_kernel(x_ptr, w_ptr, s_ptr, y_ptr, N, K, s_stride,
                     ROWS_G: tl.constexpr, SB_G: tl.constexpr,
                     BN: tl.constexpr, BK: tl.constexpr, ROUND_BF16: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    nmask = rn < N
    g = rn // ROWS_G
    srow = g * SB_G + (rn - g * ROWS_G) // 128
    acc = tl.zeros((BN,), dtype=tl.float32)
    rk = tl.arange(0, BK)
    for k0 in range(0, K, BK):
        w = tl.load(w_ptr + rn[:, None] * K + (k0 + rk)[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
        s = tl.load(s_ptr + srow * s_stride + k0 // 128, mask=nmask, other=0.0)
        w = w * s[:, None]
        if ROUND_BF16:
            w = w.to(tl.bfloat16).to(tl.float32)
        x = tl.load(x_ptr + k0 + rk).to(tl.float32)
        acc += tl.sum(w * x[None, :], 1)
    tl.store(y_ptr + rn, acc.to(y_ptr.dtype.element_ty), mask=nmask)


def fp8_gemv(x, w, s, rows_g, sb_g, out=None, BN=16, round_bf16=True, num_warps=4):
    """x [K] bf16, w [N,K] fp8e4m3, s fp32 scales [ngroups*sb_g, K/128]; row n uses scale row
    (n//rows_g)*sb_g + (n%rows_g)//128. Returns y [N] (bf16)."""
    N, K = w.shape
    if out is None:
        out = torch.empty(N, dtype=torch.bfloat16, device=x.device)
    _fp8_gemv_kernel[(triton.cdiv(N, BN),)](x, w, s, out, N, K, s.stride(0), rows_g, sb_g, BN, 128,
                                            round_bf16, num_warps=num_warps)
    return out


@triton.jit
def _fp8_gemv2_kernel(x_ptr, w_ptr, s_ptr, y_ptr, N, K, s_stride,
                      ROWS_G: tl.constexpr, SB_G: tl.constexpr,
                      BN: tl.constexpr, BK: tl.constexpr, ROUND_BF16: tl.constexpr):
    # BK multiple of 128: one scale per (row, 128-wide k block); dequant per weight (bit-exact vs BK=128 kernel's
    # weights, fp32 sum order differs only in the tree reduction)
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    nmask = rn < N
    g = rn // ROWS_G
    srow = g * SB_G + (rn - g * ROWS_G) // 128
    NS: tl.constexpr = BK // 128
    acc = tl.zeros((BN, BK), dtype=tl.float32)
    rk = tl.arange(0, BK)
    rs = tl.arange(0, NS)
    for k0 in range(0, K, BK):
        w = tl.load(w_ptr + rn[:, None] * K + (k0 + rk)[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
        s = tl.load(s_ptr + srow[:, None] * s_stride + (k0 // 128 + rs)[None, :], mask=nmask[:, None], other=0.0)
        w = tl.reshape(tl.reshape(w, (BN, NS, 128)) * s[:, :, None], (BN, BK))
        if ROUND_BF16:
            w = w.to(tl.bfloat16).to(tl.float32)
        x = tl.load(x_ptr + k0 + rk).to(tl.float32)
        acc += w * x[None, :]
    tl.store(y_ptr + rn, tl.sum(acc, 1).to(y_ptr.dtype.element_ty), mask=nmask)


def fp8_gemv2(x, w, s, rows_g, sb_g, out=None, BN=8, BK=512, round_bf16=True, num_warps=4):
    N, K = w.shape
    if out is None:
        out = torch.empty(N, dtype=torch.bfloat16, device=x.device)
    _fp8_gemv2_kernel[(triton.cdiv(N, BN),)](x, w, s, out, N, K, s.stride(0), rows_g, sb_g, BN, BK,
                                             round_bf16, num_warps=num_warps)
    return out


# ----------------------------------------------------------------------------- NVFP4 hot experts (decode)
@triton.jit
def _fp4_lut(c):
    # E2M1 decode: code c in [0,16)
    mag = c & 7
    v = tl.where(mag < 4, mag.to(tl.float32) * 0.5, tl.where(mag == 4, 2.0, tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0))))
    return tl.where(c >= 8, -v, v)


@triton.jit
def _nvfp4_gemv_kernel(x_ptr, x_stride, slot_ptr, idx_ptr, pk_ptr, bs_ptr, g2_ptr, g2_stride, y_ptr, y_stride,
                       N, K, row_off, g2_col, pk_estride, bs_estride, act_off,
                       BN: tl.constexpr, BK: tl.constexpr, ACT: tl.constexpr, XDIV: tl.constexpr = 1):
    """y[j, n] = sum_k W_e[row_off+n, k] x[j, k] for slot j with hot slot s=slot[idx[j]] (skip if -1).
    packed [E, R, K/2] uint8 (lo nibble = even k), bscale [E, R, K/16] e4m3, global g2[E, g2_col]."""
    j = tl.program_id(1)
    e = tl.load(idx_ptr + j)
    s = tl.load(slot_ptr + e)
    if s < 0:
        return
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    nmask = rn < N
    rows = row_off + rn
    g2 = tl.load(g2_ptr + s * g2_stride + g2_col)
    acc = tl.zeros((BN,), dtype=tl.float32)
    rkb = tl.arange(0, BK // 2)
    rks = tl.arange(0, BK // 16)
    pbase = pk_ptr + s.to(tl.int64) * pk_estride + rows[:, None].to(tl.int64) * (K // 2)
    sbase = bs_ptr + s.to(tl.int64) * bs_estride + rows[:, None].to(tl.int64) * (K // 16)
    xj = j // XDIV
    for k0 in range(0, K, BK):
        p = tl.load(pbase + k0 // 2 + rkb[None, :], mask=nmask[:, None], other=0)  # [BN, BK/2] u8
        lo = _fp4_lut((p & 15).to(tl.int32))
        hi = _fp4_lut((p >> 4).to(tl.int32))
        sc = tl.load(sbase + k0 // 16 + rks[None, :], mask=nmask[:, None], other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        xe = tl.load(x_ptr + xj * x_stride + k0 + 2 * rkb).to(tl.float32)
        xo = tl.load(x_ptr + xj * x_stride + k0 + 2 * rkb + 1).to(tl.float32)
        if ACT:  # x = bf16(silu(g) * u), g at x, u at x + act_off (fp32)
            ue = tl.load(x_ptr + j * x_stride + act_off + k0 + 2 * rkb)
            uo = tl.load(x_ptr + j * x_stride + act_off + k0 + 2 * rkb + 1)
            xe = _bf(xe * tl.sigmoid(xe) * ue)
            xo = _bf(xo * tl.sigmoid(xo) * uo)
        # weight = lut * scale * g2 rounded to bf16 (mimics reference dequant)
        lo3 = tl.reshape(lo, (BN, BK // 16, 8)) * sc[:, :, None] * g2
        hi3 = tl.reshape(hi, (BN, BK // 16, 8)) * sc[:, :, None] * g2
        lo3 = lo3.to(tl.bfloat16).to(tl.float32)
        hi3 = hi3.to(tl.bfloat16).to(tl.float32)
        acc += tl.sum(tl.reshape(lo3, (BN, BK // 2)) * xe[None, :], 1) + tl.sum(tl.reshape(hi3, (BN, BK // 2)) * xo[None, :], 1)
    tl.store(y_ptr + j * y_stride + rn, acc, mask=nmask)


def nvfp4_gemv(x, slot, idx, packed, bscale, g2, g2_col, row_off, N, out, BN=16, BK=256, num_warps=4, act_off=0, xdiv=1):
    """x [S, K] bf16 (per-slot input, stride may be 0 for broadcast), out [S, N] fp32.
    act_off > 0: x is fp32 [S, >= act_off + K] holding (g, u) and the input is bf16(silu(g) * u)."""
    S = idx.numel()
    K = packed.shape[2] * 2
    _nvfp4_gemv_kernel[(triton.cdiv(N, BN), S)](x, x.stride(0), slot, idx, packed, bscale, g2, g2.stride(0), out,
                                               out.stride(0), N, K, row_off, g2_col, packed.stride(0), bscale.stride(0),
                                               act_off, BN, BK, act_off > 0, xdiv, num_warps=num_warps)
    return out


# ----------------------------------------------------------------------------- decode attention
@triton.jit
def _attn_decode_kernel(q_ptr, k_ptr, v_ptr, len_ptr, o_ptr, m_ptr, l_ptr,
                        k_sh, k_ss, v_sh, v_ss, scale,
                        G: tl.constexpr, BS: tl.constexpr, NSPLIT: tl.constexpr):
    """q [NH, 192] bf16 (heads h = kv*G + i); K cache [NKV, L, 192]; V cache [NKV, L, 128].
    Valid key slots: [0, n) with n = len_ptr[0]. Partial outputs per split: o [NKV, NSPLIT, G, 128],
    m, l [NKV, NSPLIT, G]."""
    kv = tl.program_id(0)
    sp = tl.program_id(1)
    n = tl.load(len_ptr)
    SPLIT_LEN = tl.cdiv(tl.cdiv(n, NSPLIT), BS) * BS
    lo = sp * SPLIT_LEN
    hi = tl.minimum(lo + SPLIT_LEN, n)
    rg = tl.arange(0, G)
    ra = tl.arange(0, 128)
    rb = tl.arange(0, 64)
    qa = tl.load(q_ptr + (kv * G + rg)[:, None] * 192 + ra[None, :])
    qb = tl.load(q_ptr + (kv * G + rg)[:, None] * 192 + 128 + rb[None, :])
    m_i = tl.full((G,), -float("inf"), tl.float32)
    l_i = tl.zeros((G,), tl.float32)
    acc = tl.zeros((G, 128), tl.float32)
    rs = tl.arange(0, BS)
    kbase = k_ptr + kv * k_sh
    vbase = v_ptr + kv * v_sh
    for s0 in range(lo, hi, BS):
        sm = (s0 + rs) < hi
        ka = tl.load(kbase + (s0 + rs)[:, None] * k_ss + ra[None, :], mask=sm[:, None], other=0.0)
        kb = tl.load(kbase + (s0 + rs)[:, None] * k_ss + 128 + rb[None, :], mask=sm[:, None], other=0.0)
        sc = tl.dot(qa, tl.trans(ka)) + tl.dot(qb, tl.trans(kb))  # [G, BS] fp32
        # reference: bf16 matmul output then .float() * scale
        sc = sc.to(tl.bfloat16).to(tl.float32) * scale
        sc = tl.where(sm[None, :], sc, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(vbase + (s0 + rs)[:, None] * v_ss + ra[None, :], mask=sm[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    base = (kv * NSPLIT + sp) * G
    tl.store(o_ptr + (base + rg)[:, None] * 128 + ra[None, :], acc)
    tl.store(m_ptr + base + rg, m_i)
    tl.store(l_ptr + base + rg, l_i)


@triton.jit
def _attn_combine_kernel(o_ptr, m_ptr, l_ptr, sink_ptr, out_ptr, G: tl.constexpr, NSPLIT: tl.constexpr,
                         HAS_SINK: tl.constexpr):
    h = tl.program_id(0)  # head
    kv = h // G
    i = h % G
    rsp = tl.arange(0, NSPLIT)
    idx = (kv * NSPLIT + rsp) * G + i
    m = tl.load(m_ptr + idx)
    l = tl.load(l_ptr + idx)
    M = tl.max(m, 0)
    if HAS_SINK:
        sk = tl.load(sink_ptr + h).to(tl.float32)
        M = tl.maximum(M, sk)
    w = tl.where(m == -float("inf"), 0.0, tl.exp(m - M))
    L = tl.sum(l * w, 0)
    if HAS_SINK:
        L += tl.exp(sk - M)
    ra = tl.arange(0, 128)
    o = tl.load(o_ptr + idx[:, None] * 128 + ra[None, :])
    o = tl.sum(o * w[:, None], 0) / L
    tl.store(out_ptr + h * 128 + ra, o.to(tl.bfloat16))


class DecodeAttn:
    """Scratch for split-KV decode attention on one device."""

    def __init__(self, dev, nkv=8, g=16, nsplit=16):
        self.nkv, self.g, self.nsplit = nkv, g, nsplit
        self.o = torch.empty(nkv * nsplit * g, 128, dtype=torch.float32, device=dev)
        self.m = torch.empty(nkv * nsplit * g, dtype=torch.float32, device=dev)
        self.l = torch.empty(nkv * nsplit * g, dtype=torch.float32, device=dev)

    def __call__(self, q, kc, vc, n_dev, sink, out, scale, BS=32):
        """q [NH,192] bf16 contiguous; kc [NKV,L,192]; vc [NKV,L,128]; n_dev int32 [1] (valid slots)."""
        ns = self.nsplit
        _attn_decode_kernel[(self.nkv, ns)](q, kc, vc, n_dev, self.o, self.m, self.l,
                                            kc.stride(0), kc.stride(1), vc.stride(0), vc.stride(1),
                                            scale, self.g, BS, ns, num_warps=4)
        _attn_combine_kernel[(self.nkv * self.g,)](self.o, self.m, self.l, sink if sink is not None else self.m,
                                                   out, self.g, ns, sink is not None, num_warps=1)
        return out


# ----------------------------------------------------------------------------- fused small ops (decode)
@triton.jit
def _add_rmsnorm_kernel(x_ptr, d_ptr, w_ptr, xo_ptr, h_ptr, N, eps, HAS_D: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.arange(0, BLOCK)
    m = r < N
    x = tl.load(x_ptr + r, mask=m, other=0.0).to(tl.float32)
    if HAS_D:
        d = tl.load(d_ptr + r, mask=m, other=0.0).to(tl.float32)
        d = _bf(d)  # no-op for bf16 d; fp32 d (TP all-reduced sum) is rounded like the reference's bf16 output
        x = _bf(x + d)
        tl.store(xo_ptr + r, x.to(tl.bfloat16), mask=m)
    ms = tl.sum(x * x, 0) / N
    y = (x * tl.math.rsqrt(ms + eps)).to(tl.bfloat16).to(tl.float32)
    w = tl.load(w_ptr + r, mask=m, other=0.0).to(tl.float32)
    tl.store(h_ptr + r, (w * y).to(tl.bfloat16), mask=m)


def add_rmsnorm(x, w, eps, d=None, xo=None, h=None):
    """xo = bf16(x + d) (if d), h = rmsnorm(xo) exactly like the reference (bf16 roundings)."""
    N = x.numel()
    if h is None:
        h = torch.empty_like(x)
    if d is not None and xo is None:
        xo = torch.empty_like(x)
    _add_rmsnorm_kernel[(1,)](x, d if d is not None else x, w, xo if xo is not None else x, h, N, eps,
                              d is not None, triton.next_power_of_2(N), num_warps=16)
    return (xo if d is not None else x), h


@triton.jit
def _qkv_post_kernel(qkv_ptr, cos_ptr, sin_ptr, pos_ptr, slot_ptr, q_ptr, kc_ptr, vc_ptr, kc_sh, kc_ss, vc_sh, vc_ss,
                     vscale, ROWS_G: tl.constexpr, G: tl.constexpr, HD: tl.constexpr, VD: tl.constexpr,
                     ROPE: tl.constexpr):
    """one program per kv group g: rows [G q heads | k | v] of width HD (v: VD). Rope on first ROPE dims
    (bf16 op-by-op like the reference), q -> q_ptr [NH, HD], k/v -> caches at slot."""
    g = tl.program_id(0)
    pos = tl.load(pos_ptr)
    slot = tl.load(slot_ptr)
    RH: tl.constexpr = ROPE // 2
    rr = tl.arange(0, ROPE)
    rot_src = tl.where(rr < RH, rr + RH, rr - RH)
    sign = tl.where(rr < RH, -1.0, 1.0)
    cos = tl.load(cos_ptr + pos * ROPE + rr).to(tl.float32)
    sin = tl.load(sin_ptr + pos * ROPE + rr).to(tl.float32)
    rh = tl.arange(0, 2 * G)  # rows 0..G-1 q heads, G = k
    hm = rh <= G
    base = qkv_ptr + g * ROWS_G
    r = tl.load(base + rh[:, None] * HD + rr[None, :], mask=hm[:, None], other=0.0).to(tl.float32)
    rs = tl.load(base + rh[:, None] * HD + rot_src[None, :], mask=hm[:, None], other=0.0).to(tl.float32) * sign[None, :]
    o = _bf(_bf(r * cos[None, :]) + _bf(rs * sin[None, :]))
    NR: tl.constexpr = HD - ROPE
    rn = tl.arange(0, NR)
    rest = tl.load(base + rh[:, None] * HD + ROPE + rn[None, :], mask=hm[:, None], other=0.0)
    isq = rh < G
    qrow = g * G + rh
    tl.store(q_ptr + qrow[:, None] * HD + rr[None, :], o.to(tl.bfloat16), mask=isq[:, None])
    tl.store(q_ptr + qrow[:, None] * HD + ROPE + rn[None, :], rest, mask=isq[:, None])
    isk = rh == G
    kb = kc_ptr + g * kc_sh + slot * kc_ss
    tl.store(kb + rh[:, None] * 0 + rr[None, :], o.to(tl.bfloat16), mask=isk[:, None])
    tl.store(kb + rh[:, None] * 0 + ROPE + rn[None, :], rest, mask=isk[:, None])
    rv = tl.arange(0, VD)
    v = tl.load(base + (G + 1) * HD + rv).to(tl.float32)
    v = (v * vscale).to(tl.bfloat16)
    tl.store(vc_ptr + g * vc_sh + slot * vc_ss + rv, v)


def qkv_post(qkv, cos, sin, pos, slot, q_out, kc, vc, vscale, nkv, rows_g, g, hd, vd, rope):
    _qkv_post_kernel[(nkv,)](qkv, cos, sin, pos, slot, q_out, kc, vc, kc.stride(0), kc.stride(1), vc.stride(0),
                             vc.stride(1), vscale, rows_g, g, hd, vd, rope, num_warps=4)


@triton.jit
def _route_kernel(lg_ptr, b_ptr, idx_ptr, wt_ptr, NE, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.arange(0, BLOCK)
    m = r < NE
    s = tl.sigmoid(tl.load(lg_ptr + r, mask=m, other=0.0))
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
    tl.store(idx_ptr + rk, iv)
    tl.store(wt_ptr + rk, wv / (tot + 1e-20))


def route(logits, bias, idx_out, wt_out, topk=8):
    NE = logits.numel()
    _route_kernel[(1,)](logits, bias, idx_out, wt_out, NE, topk, triton.next_power_of_2(NE), num_warps=4)


# ----------------------------------------------------------------------------- bf16 GEMV, fp32 out (TP partials)
@triton.jit
def _bf16_gemv_kernel(x_ptr, w_ptr, y_ptr, N, K, BN: tl.constexpr, BK: tl.constexpr, ROUND: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    nmask = rn < N
    rk = tl.arange(0, BK)
    acc = tl.zeros((BN, BK), dtype=tl.float32)
    for k0 in range(0, K, BK):
        w = tl.load(w_ptr + rn[:, None].to(tl.int64) * K + (k0 + rk)[None, :], mask=nmask[:, None], other=0.0)
        x = tl.load(x_ptr + k0 + rk)
        acc += w.to(tl.float32) * x.to(tl.float32)[None, :]
    y = tl.sum(acc, 1)
    if ROUND:
        y = _bf(y)
    tl.store(y_ptr + rn, y, mask=nmask)


def bf16_gemv(x, w, out, BN=8, BK=512, round_out=False, num_warps=4):
    """out[N] fp32 = w[N,K] bf16 @ x[K] bf16 (K % BK == 0)."""
    N, K = w.shape
    assert K % BK == 0
    _bf16_gemv_kernel[(triton.cdiv(N, BN),)](x, w, out, N, K, BN, BK, round_out, num_warps=num_warps)
    return out


@triton.jit
def _gate_gemv_kernel(x_ptr, w_ptr, y_ptr, N, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BN, BK), dtype=tl.float32)
    for k0 in range(0, K, BK):
        w = tl.load(w_ptr + rn[:, None] * K + (k0 + rk)[None, :])
        x = tl.load(x_ptr + k0 + rk).to(tl.float32)
        acc += w * x[None, :]
    tl.store(y_ptr + rn, tl.sum(acc, 1))


def gate_gemv(x, w, out, BN=2, BK=1024, num_warps=4):
    """out[N] fp32 = w[N,K] fp32 @ float(x[K] bf16). N % BN == 0, K % BK == 0."""
    N, K = w.shape
    _gate_gemv_kernel[(N // BN,)](x, w, out, N, K, BN, BK, num_warps=num_warps)
    return out

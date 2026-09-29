// Heterogeneous-bitrate batch-1 MoE decode for MiMo-V2.6-Pro-EXL3 cold experts.
// Reuses exllamav3's gemv_tile (exl3_moe_coop_kernel.cuh) but dispatches the bitrate per (expert, piece)
// at runtime, so experts whose gate/up/down pieces have different K (1.5/2/2.5) can run in one launch.
// Four graph-safe launches per layer, routing read on device:
//   gu:   (slot, proj, col-group) blocks: x*suh -> Hadamard (smem) -> trellis GEMV -> raw fp32 [slot, proj, 2048]
//   act:  (slot, down-piece) blocks: out-Hadamard*svh on g,u, silu(g)*u clamp -> half, *down suh -> Hadamard -> half
//   down: (slot, piece, col-group) blocks: trellis GEMV -> raw fp32 [slot, H] at the piece's column offset
//   out:  per 128-col chunk: out-Hadamard*svh (cold) or hot result, bf16 round, routing-weighted sum -> bf16 [H]
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "quant/exl3_moe_coop_kernel.cuh"

using namespace exl3_moe_coop_ns;

#define HDIM 6144
#define NPIECE 3

// Table layouts (per layer, device):
//   gu_ptr  int64 [NE, 2, 3]  (trellis, suh, svh) for gate/up; trellis 0 = not cold (skip)
//   gu_k2   int32 [NE, 2]     2*K
//   dn_ptr  int64 [NE, NPIECE, 3]
//   dn_meta int32 [NE, NPIECE, 3]  (2*K, n, col); n = 0: absent piece

template <bool WIDE>
__device__ __forceinline__ void tile_dispatch(int k2, const uint32_t* B32, const half2* A2, const int* rows,
                                              void* C, int k_begin, int k_end, int ntiles, int group, float* sh_red, uint32_t* sh_stage)
{
    switch (k2)
    {
        case 3: gemv_tile<1, 2, WIDE, true >(B32, A2, 0, rows, 1, C, 0, true, k_begin, k_end, ntiles, group, sh_red, sh_stage); break;
        case 4: gemv_tile<2, 2, WIDE, false>(B32, A2, 0, rows, 1, C, 0, true, k_begin, k_end, ntiles, group, sh_red, sh_stage); break;
        case 5: gemv_tile<2, 2, WIDE, true >(B32, A2, 0, rows, 1, C, 0, true, k_begin, k_end, ntiles, group, sh_red, sh_stage); break;
        case 6: gemv_tile<3, 2, WIDE, false>(B32, A2, 0, rows, 1, C, 0, true, k_begin, k_end, ntiles, group, sh_red, sh_stage); break;
        default: break;
    }
}

template <bool WIDE>
__global__ __launch_bounds__(THREADS)
void moe_gu_kernel(const __nv_bfloat16* __restrict__ x, const int64_t* __restrict__ sel,
                   const int64_t* __restrict__ gu_ptr, const int* __restrict__ gu_k2, float* __restrict__ gu_raw, int IDIM,
                   int ks_n, size_t ks_stride)
{
    constexpr int TCOLS = tile_cols<WIDE>();
    const int NG = IDIM / TCOLS;
    const int nb = gridDim.x / ks_n;
    const int ks = blockIdx.x / nb;
    const int bid = blockIdx.x % nb;
    __shared__ __align__(16) half sh_A[HDIM];
    __shared__ __align__(16) float sh_red[WK * ROWS * COLS];
    __shared__ __align__(16) uint32_t sh_stage[WK * STAGE_WORDS];
    __shared__ int sh_rows[ROWS];

    const int group = bid % NG;
    const int j = (bid / NG) % 2;
    const int s = bid / (NG * 2);
    const int64_t e = sel[s];
    const int64_t* pt = gu_ptr + (e * 2 + j) * 3;
    const uint32_t* B32 = (const uint32_t*) pt[0];
    if (!B32) return;
    const half* suh = (const half*) pt[1];
    const int k2 = gu_k2[e * 2 + j];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int c = warp; c < HDIM / 128; c += WK)
    {
        const int col = c * 128 + lane * 4;
        float v0 = __bfloat162float(x[col]), v1 = __bfloat162float(x[col + 1]);
        float v2 = __bfloat162float(x[col + 2]), v3 = __bfloat162float(x[col + 3]);
        // reference: x -> half, then had_r_128(x * suh)
        v0 = __half2float(__float2half_rn(v0)); v1 = __half2float(__float2half_rn(v1));
        v2 = __half2float(__float2half_rn(v2)); v3 = __half2float(__float2half_rn(v3));
        scale_h4(suh + col, v0, v1, v2, v3);
        had128(v0, v1, v2, v3, lane);
        store_h4(sh_A + col, v0, v1, v2, v3);
    }
    if (threadIdx.x == 0) sh_rows[0] = 0;
    __syncthreads();
    float* C = gu_raw + ks * ks_stride + (size_t) (s * 2 + j) * IDIM;
    const int K16 = HDIM / 16, kc = K16 / ks_n;
    tile_dispatch<WIDE>(k2, B32, (const half2*) sh_A, sh_rows, C, ks * kc, (ks + 1) * kc, IDIM / 16, group, sh_red, sh_stage);
}

// grid: TOPK * NPIECE blocks, 512 threads (16 warps = 16 chunks of 128 over IDIM)
__global__ __launch_bounds__(512)
void moe_act_kernel(const int64_t* __restrict__ sel, const int64_t* __restrict__ gu_ptr,
                    const int64_t* __restrict__ dn_ptr, const int* __restrict__ dn_meta,
                    const float* __restrict__ gu_raw, half* __restrict__ act, float limit, int IDIM,
                    int ks_n, size_t ks_stride)
{
    const int s = blockIdx.x / NPIECE;
    const int p = blockIdx.x % NPIECE;
    const int64_t e = sel[s];
    const int64_t* dp = dn_ptr + (e * NPIECE + p) * 3;
    if (!dp[0] || dn_meta[(e * NPIECE + p) * 3 + 1] == 0) return;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int col = warp * 128 + lane * 4; col < IDIM; col += 16 * 128) {
    const int64_t* gp = gu_ptr + (e * 2 + 0) * 3;
    const int64_t* up = gu_ptr + (e * 2 + 1) * 3;
    float g0, g1, g2, g3, u0, u1, u2, u3;
    const float* gr = gu_raw + (size_t) (s * 2) * IDIM + col;
    const float* ur = gu_raw + (size_t) (s * 2 + 1) * IDIM + col;
    g0 = gr[0]; g1 = gr[1]; g2 = gr[2]; g3 = gr[3];
    u0 = ur[0]; u1 = ur[1]; u2 = ur[2]; u3 = ur[3];
    for (int q = 1; q < ks_n; ++q)
    {
        const float* gq = gr + q * ks_stride; const float* uq = ur + q * ks_stride;
        g0 += gq[0]; g1 += gq[1]; g2 += gq[2]; g3 += gq[3];
        u0 += uq[0]; u1 += uq[1]; u2 += uq[2]; u3 += uq[3];
    }
    had128(g0, g1, g2, g3, lane);
    scale_h4(((const half*) gp[2]) + col, g0, g1, g2, g3);
    had128(u0, u1, u2, u3, lane);
    scale_h4(((const half*) up[2]) + col, u0, u1, u2, u3);
    auto a = [&] (float g, float u) {
        float v = act_silu(g) * u;
        v = fminf(fmaxf(v, -limit), limit);
        return __half2float(__float2half_rn(v));
    };
    float a0 = a(g0, u0), a1 = a(g1, u1), a2 = a(g2, u2), a3 = a(g3, u3);
    scale_h4(((const half*) dp[1]) + col, a0, a1, a2, a3);
    had128(a0, a1, a2, a3, lane);
    store_h4(act + (size_t) (s * NPIECE + p) * IDIM + col, a0, a1, a2, a3);
    }
}

template <bool WIDE>
__global__ __launch_bounds__(THREADS)
void moe_down_kernel(const int64_t* __restrict__ sel, const int64_t* __restrict__ dn_ptr, const int* __restrict__ dn_meta,
                     const half* __restrict__ act, float* __restrict__ d_raw, int ng, int IDIM, int ks_n, size_t ks_stride)
{
    constexpr int TCOLS = tile_cols<WIDE>();
    const int nb = gridDim.x / ks_n;
    const int ks = blockIdx.x / nb;
    const int bid = blockIdx.x % nb;
    __shared__ __align__(16) float sh_red[WK * ROWS * COLS];
    __shared__ __align__(16) uint32_t sh_stage[WK * STAGE_WORDS];
    __shared__ int sh_rows[ROWS];
    const int group = bid % ng;
    const int p = (bid / ng) % NPIECE;
    const int s = bid / (ng * NPIECE);
    const int64_t e = sel[s];
    const int64_t* dp = dn_ptr + (e * NPIECE + p) * 3;
    const uint32_t* B32 = (const uint32_t*) dp[0];
    if (!B32) return;
    const int* m = dn_meta + (e * NPIECE + p) * 3;
    const int n = m[1];
    if (group * TCOLS >= n) return;
    if (threadIdx.x == 0) sh_rows[0] = 0;
    __syncthreads();
    float* C = d_raw + ks * ks_stride + (size_t) s * HDIM + m[2];
    const half2* A2 = (const half2*) (act + (size_t) (s * NPIECE + p) * IDIM);
    const int K16 = IDIM / 16, kc = K16 / ks_n;
    tile_dispatch<WIDE>(m[0], B32, A2, sh_rows, C, ks * kc, (ks + 1) * kc, n / 16, group, sh_red, sh_stage);
}

// grid: HDIM/128 blocks, TOPK warps. hot_y [TOPK, HDIM] fp32 (hot result per slot, may be null), is_hot bool [NE]
__global__ __launch_bounds__(256)
void moe_out_kernel(const int64_t* __restrict__ sel, const float* __restrict__ wt, const int64_t* __restrict__ dn_ptr,
                    const int* __restrict__ dn_meta, const float* __restrict__ d_raw, const float* __restrict__ hot_y,
                    const bool* __restrict__ is_hot, const __nv_bfloat16* __restrict__ res,
                    void* __restrict__ out_, int topk, int round_out, int ks_n, size_t ks_stride)
{
    __shared__ float part[8][128];
    const int c = blockIdx.x;
    const int s = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int col = c * 128 + lane * 4;
    float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
    if (s < topk)
    {
        const int64_t e = sel[s];
        const float w = wt[s];
        if (hot_y && is_hot[e])
        {
            const float* h = hot_y + (size_t) s * HDIM + col;
            v0 = h[0]; v1 = h[1]; v2 = h[2]; v3 = h[3];
        }
        else
        {
            int p = 0, c0 = 0;
            for (int q = 0; q < NPIECE; ++q)
            {
                const int* m = dn_meta + (e * NPIECE + q) * 3;
                if (m[1] > 0 && col >= m[2] && col < m[2] + m[1]) { p = q; c0 = m[2]; }
            }
            const float* d = d_raw + (size_t) s * HDIM + col;
            v0 = d[0]; v1 = d[1]; v2 = d[2]; v3 = d[3];
            for (int q = 1; q < ks_n; ++q)
            {
                const float* dq = d + q * ks_stride;
                v0 += dq[0]; v1 += dq[1]; v2 += dq[2]; v3 += dq[3];
            }
            had128(v0, v1, v2, v3, lane);
            scale_h4(((const half*) dn_ptr[(e * NPIECE + p) * 3 + 2]) + (col - c0), v0, v1, v2, v3);
        }
        // reference: expert output bf16, then fp32 weighted sum (TP partial: no rounding, summed across ranks)
        if (round_out)
        {
            v0 = __bfloat162float(__float2bfloat16_rn(v0));
            v1 = __bfloat162float(__float2bfloat16_rn(v1));
            v2 = __bfloat162float(__float2bfloat16_rn(v2));
            v3 = __bfloat162float(__float2bfloat16_rn(v3));
        }
        v0 *= w; v1 *= w; v2 *= w; v3 *= w;
    }
    part[s][lane * 4 + 0] = v0; part[s][lane * 4 + 1] = v1; part[s][lane * 4 + 2] = v2; part[s][lane * 4 + 3] = v3;
    __syncthreads();
    if (threadIdx.x < 128)
    {
        float acc = 0.f;
        for (int q = 0; q < 8; ++q) acc += part[q][threadIdx.x];
        if (!round_out) { ((float*) out_)[c * 128 + threadIdx.x] = acc; return; }
        if (res) acc = __bfloat162float(__float2bfloat16_rn(acc)) + __bfloat162float(res[c * 128 + threadIdx.x]);
        ((__nv_bfloat16*) out_)[c * 128 + threadIdx.x] = __float2bfloat16_rn(acc);
    }
}

static int g_wide_gu = 1, g_wide_dn = 0, g_ks_gu = 1, g_ks_dn = 1;

void set_wide(int gu, int dn) { g_wide_gu = gu; g_wide_dn = dn; }
void set_ksplit(int gu, int dn) { g_ks_gu = gu; g_ks_dn = dn; }

// x [H] bf16, sel [TOPK] int64, wt [TOPK] fp32; scratch gu_raw [TOPK*2*I] f32, act [TOPK*NPIECE*I] half,
// d_raw [TOPK*H] f32; hot_y optional [TOPK, H] f32; out [H] bf16
void moe_cold(torch::Tensor x, torch::Tensor sel, torch::Tensor wt,
              torch::Tensor gu_ptr, torch::Tensor gu_k2, torch::Tensor dn_ptr, torch::Tensor dn_meta, int maxn,
              torch::Tensor gu_raw, torch::Tensor act, torch::Tensor d_raw,
              c10::optional<torch::Tensor> hot_y, torch::Tensor is_hot, c10::optional<torch::Tensor> res,
              torch::Tensor out, double limit, int64_t I)
{
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int topk = (int) sel.numel();
    TORCH_CHECK(topk <= 8);
    const auto* xp = (const __nv_bfloat16*) x.data_ptr();
    const auto* sp = (const int64_t*) sel.data_ptr();
    const auto* gp = (const int64_t*) gu_ptr.data_ptr();
    const auto* gk = (const int*) gu_k2.data_ptr();
    const auto* dp = (const int64_t*) dn_ptr.data_ptr();
    const auto* dm = (const int*) dn_meta.data_ptr();
    float* gr = (float*) gu_raw.data_ptr();
    half* ap = (half*) act.data_ptr();
    float* dr = (float*) d_raw.data_ptr();
    const int ksg = g_ks_gu, ksd = g_ks_dn;
    const size_t gstr = (size_t) topk * 2 * I, dstr = (size_t) topk * HDIM;
    TORCH_CHECK(gu_raw.numel() >= (int64_t) (ksg * gstr) && d_raw.numel() >= (int64_t) (ksd * dstr), "moe_cold: ksplit scratch too small");
    TORCH_CHECK((HDIM / 16) % ksg == 0 && (I / 16) % ksd == 0, "moe_cold: bad ksplit");
    if (g_wide_gu) moe_gu_kernel<true><<<ksg * topk * 2 * ((int) I / tile_cols<true>()), THREADS, 0, st>>>(xp, sp, gp, gk, gr, (int) I, ksg, gstr);
    else           moe_gu_kernel<false><<<ksg * topk * 2 * ((int) I / tile_cols<false>()), THREADS, 0, st>>>(xp, sp, gp, gk, gr, (int) I, ksg, gstr);
    moe_act_kernel<<<topk * NPIECE, 512, 0, st>>>(sp, gp, dp, dm, gr, ap, (float) limit, (int) I, ksg, gstr);
    if (g_wide_dn)
    {
        const int ng = CEIL_DIVIDE(maxn, tile_cols<true>());
        moe_down_kernel<true><<<ksd * topk * NPIECE * ng, THREADS, 0, st>>>(sp, dp, dm, ap, dr, ng, (int) I, ksd, dstr);
    }
    else
    {
        const int ng = CEIL_DIVIDE(maxn, tile_cols<false>());
        moe_down_kernel<false><<<ksd * topk * NPIECE * ng, THREADS, 0, st>>>(sp, dp, dm, ap, dr, ng, (int) I, ksd, dstr);
    }
    moe_out_kernel<<<HDIM / 128, 256, 0, st>>>(sp, (const float*) wt.data_ptr(), dp, dm, dr,
                                               hot_y ? (const float*) hot_y->data_ptr() : nullptr,
                                               (const bool*) is_hot.data_ptr(),
                                               res ? (const __nv_bfloat16*) res->data_ptr() : nullptr,
                                               out.data_ptr(), topk, out.scalar_type() == at::kBFloat16 ? 1 : 0, ksd, dstr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// ============================================================================ M-token (spec-verify) path
// Slots s = tok * tpk + k (tpk = top-k). Cold slots are grouped into runs of one expert (<= ROWS slots) so each
// expert's trellis weights are decoded once per run and multiplied against all its tokens (m16 MMA rows).
// runs: int [2 + (S+1) + S]: [n_runs, -, run_start[0..n_runs], order[0..n_active)]
#define MAXS 64
__global__ void moe_runs_kernel(const int64_t* __restrict__ sel, const int64_t* __restrict__ gu_ptr, int S, int* runs)
{
    __shared__ int e_of[MAXS];
    __shared__ int ord[MAXS];
    const int t = threadIdx.x;
    if (t < S)
    {
        const int64_t e = sel[t];
        e_of[t] = gu_ptr[(e * 2) * 3] ? (int) e : -1;
    }
    __syncthreads();
    if (t < S && e_of[t] >= 0)
    {
        const int key = (e_of[t] << 8) | t;
        int rank = 0;
        for (int u = 0; u < S; ++u) rank += (e_of[u] >= 0 && ((e_of[u] << 8) | u) < key) ? 1 : 0;
        ord[rank] = t;
    }
    __syncthreads();
    if (t == 0)
    {
        int na = 0;
        for (int u = 0; u < S; ++u) na += e_of[u] >= 0 ? 1 : 0;
        int* run_start = runs + 2;
        int* order = runs + 2 + S + 1;
        int nr = 0, first = 0;
        for (int i = 0; i < na; ++i)
        {
            const bool st = i == 0 || e_of[ord[i]] != e_of[ord[i - 1]] || (i - first) >= ROWS;
            if (st) { run_start[nr++] = i; first = i; }
            order[i] = ord[i];
        }
        run_start[nr] = na;
        runs[0] = nr;
    }
}

__device__ __forceinline__ bool read_run_m(const int* runs, int S, int run_idx, int* rows, int& nrows, int64_t& e,
                                           const int64_t* sel)
{
    const int n_runs = runs[0];
    if (run_idx >= n_runs) return false;
    const int start = runs[2 + run_idx];
    nrows = runs[2 + run_idx + 1] - start;
    if (threadIdx.x < nrows) rows[threadIdx.x] = runs[2 + S + 1 + start + threadIdx.x];
    e = sel[runs[2 + S + 1 + start]];
    __syncthreads();
    return true;
}

// rot: grid S*2 blocks (slot, proj), 512 threads: xr[slot, proj, :] = had128(half(x[tok]) * suh)
__global__ __launch_bounds__(512)
void moe_rot_kernel(const __nv_bfloat16* __restrict__ x, const int64_t* __restrict__ sel,
                    const int64_t* __restrict__ gu_ptr, half* __restrict__ xr, int tpk)
{
    const int s = blockIdx.x / 2, j = blockIdx.x % 2;
    const int64_t e = sel[s];
    const int64_t* pt = gu_ptr + (e * 2 + j) * 3;
    if (!pt[0]) return;
    const half* suh = (const half*) pt[1];
    const __nv_bfloat16* xs = x + (size_t) (s / tpk) * HDIM;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int c = warp; c < HDIM / 128; c += 16)
    {
        const int col = c * 128 + lane * 4;
        float v0 = __half2float(__float2half_rn(__bfloat162float(xs[col])));
        float v1 = __half2float(__float2half_rn(__bfloat162float(xs[col + 1])));
        float v2 = __half2float(__float2half_rn(__bfloat162float(xs[col + 2])));
        float v3 = __half2float(__float2half_rn(__bfloat162float(xs[col + 3])));
        scale_h4(suh + col, v0, v1, v2, v3);
        had128(v0, v1, v2, v3, lane);
        store_h4(xr + ((size_t) s * 2 + j) * HDIM + col, v0, v1, v2, v3);
    }
}

template <bool WIDE>
__device__ __forceinline__ void tile_dispatch_m(int k2, const uint32_t* B32, const half2* A2, size_t as2, const int* rows,
                                                int nrows, void* C, size_t cs, int k_begin, int k_end, int ntiles, int group,
                                                float* sh_red, uint32_t* sh_stage)
{
    switch (k2)
    {
        case 3: gemv_tile<1, 2, WIDE, true >(B32, A2, as2, rows, nrows, C, cs, true, k_begin, k_end, ntiles, group, sh_red, sh_stage); break;
        case 4: gemv_tile<2, 2, WIDE, false>(B32, A2, as2, rows, nrows, C, cs, true, k_begin, k_end, ntiles, group, sh_red, sh_stage); break;
        case 5: gemv_tile<2, 2, WIDE, true >(B32, A2, as2, rows, nrows, C, cs, true, k_begin, k_end, ntiles, group, sh_red, sh_stage); break;
        case 6: gemv_tile<3, 2, WIDE, false>(B32, A2, as2, rows, nrows, C, cs, true, k_begin, k_end, ntiles, group, sh_red, sh_stage); break;
        default: break;
    }
}

// gu: grid S * 2 * NG (run, proj, group); A rows = rotated inputs of the run's slots
template <bool WIDE>
__global__ __launch_bounds__(THREADS)
void moe_gu_m_kernel(const int64_t* __restrict__ sel, const int* __restrict__ runs, int S, const half* __restrict__ xr,
                     const int64_t* __restrict__ gu_ptr, const int* __restrict__ gu_k2, float* __restrict__ gu_raw, int IDIM)
{
    constexpr int TCOLS = tile_cols<WIDE>();
    const int NG = IDIM / TCOLS;
    __shared__ __align__(16) float sh_red[WK * ROWS * COLS];
    __shared__ __align__(16) uint32_t sh_stage[WK * STAGE_WORDS];
    __shared__ int sh_rows[ROWS];
    const int group = blockIdx.x % NG;
    const int j = (blockIdx.x / NG) % 2;
    const int r = blockIdx.x / (NG * 2);
    int nrows; int64_t e;
    if (!read_run_m(runs, S, r, sh_rows, nrows, e, sel)) return;
    const int64_t* pt = gu_ptr + (e * 2 + j) * 3;
    const uint32_t* B32 = (const uint32_t*) pt[0];
    const int k2 = gu_k2[e * 2 + j];
    // row s of A at xr + (s*2 + j)*HDIM  ->  A2 base = xr + j*HDIM, stride 2*HDIM halfs = HDIM half2
    // row s of C at gu_raw + (s*2 + j)*IDIM -> base gu_raw + j*IDIM, stride 2*IDIM
    tile_dispatch_m<WIDE>(k2, B32, (const half2*) (xr + (size_t) j * HDIM), HDIM, sh_rows, nrows,
                          gu_raw + (size_t) j * IDIM, 2 * IDIM, 0, HDIM / 16, IDIM / 16, group, sh_red, sh_stage);
}

// down: grid S * NPIECE * ng (run, piece, group)
template <bool WIDE>
__global__ __launch_bounds__(THREADS)
void moe_down_m_kernel(const int64_t* __restrict__ sel, const int* __restrict__ runs, int S,
                       const int64_t* __restrict__ dn_ptr, const int* __restrict__ dn_meta,
                       const half* __restrict__ act, float* __restrict__ d_raw, int ng, int IDIM)
{
    constexpr int TCOLS = tile_cols<WIDE>();
    __shared__ __align__(16) float sh_red[WK * ROWS * COLS];
    __shared__ __align__(16) uint32_t sh_stage[WK * STAGE_WORDS];
    __shared__ int sh_rows[ROWS];
    const int group = blockIdx.x % ng;
    const int p = (blockIdx.x / ng) % NPIECE;
    const int r = blockIdx.x / (ng * NPIECE);
    int nrows; int64_t e;
    if (!read_run_m(runs, S, r, sh_rows, nrows, e, sel)) return;
    const int64_t* dp = dn_ptr + (e * NPIECE + p) * 3;
    const uint32_t* B32 = (const uint32_t*) dp[0];
    if (!B32) return;
    const int* m = dn_meta + (e * NPIECE + p) * 3;
    const int n = m[1];
    if (group * TCOLS >= n) return;
    // row s of A at act + (s*NPIECE + p)*IDIM; row s of C at d_raw + s*HDIM + m[2]
    tile_dispatch_m<WIDE>(m[0], B32, (const half2*) (act + (size_t) p * IDIM), (size_t) NPIECE * IDIM / 2, sh_rows, nrows,
                          d_raw + m[2], HDIM, 0, IDIM / 16, n / 16, group, sh_red, sh_stage);
}

// out: grid (HDIM/128, ntok), 256 threads (tpk warps used)
__global__ __launch_bounds__(256)
void moe_out_m_kernel(const int64_t* __restrict__ sel, const float* __restrict__ wt, const int64_t* __restrict__ dn_ptr,
                      const int* __restrict__ dn_meta, const float* __restrict__ d_raw, const float* __restrict__ hot_y,
                      const bool* __restrict__ is_hot, const __nv_bfloat16* __restrict__ res,
                      void* __restrict__ out_, int tpk, int round_out)
{
    __shared__ float part[8][128];
    const int c = blockIdx.x, tok = blockIdx.y;
    const int k = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int s = tok * tpk + k;
    const int col = c * 128 + lane * 4;
    float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
    if (k < tpk)
    {
        const int64_t e = sel[s];
        const float w = wt[s];
        if (hot_y && is_hot[e])
        {
            const float* h = hot_y + (size_t) s * HDIM + col;
            v0 = h[0]; v1 = h[1]; v2 = h[2]; v3 = h[3];
        }
        else
        {
            int p = 0, c0 = 0;
            for (int q = 0; q < NPIECE; ++q)
            {
                const int* m = dn_meta + (e * NPIECE + q) * 3;
                if (m[1] > 0 && col >= m[2] && col < m[2] + m[1]) { p = q; c0 = m[2]; }
            }
            const float* d = d_raw + (size_t) s * HDIM + col;
            v0 = d[0]; v1 = d[1]; v2 = d[2]; v3 = d[3];
            had128(v0, v1, v2, v3, lane);
            scale_h4(((const half*) dn_ptr[(e * NPIECE + p) * 3 + 2]) + (col - c0), v0, v1, v2, v3);
        }
        if (round_out)
        {
            v0 = __bfloat162float(__float2bfloat16_rn(v0));
            v1 = __bfloat162float(__float2bfloat16_rn(v1));
            v2 = __bfloat162float(__float2bfloat16_rn(v2));
            v3 = __bfloat162float(__float2bfloat16_rn(v3));
        }
        v0 *= w; v1 *= w; v2 *= w; v3 *= w;
    }
    part[k][lane * 4 + 0] = v0; part[k][lane * 4 + 1] = v1; part[k][lane * 4 + 2] = v2; part[k][lane * 4 + 3] = v3;
    __syncthreads();
    if (threadIdx.x < 128)
    {
        float acc = 0.f;
        for (int q = 0; q < 8; ++q) acc += part[q][threadIdx.x];
        const size_t o = (size_t) tok * HDIM + c * 128 + threadIdx.x;
        if (!round_out) { ((float*) out_)[o] = acc; return; }
        if (res) acc = __bfloat162float(__float2bfloat16_rn(acc)) + __bfloat162float(res[o]);
        ((__nv_bfloat16*) out_)[o] = __float2bfloat16_rn(acc);
    }
}

// x [M, H] bf16, sel/wt [M*tpk]; scratch: runs int [2 + 2S + 1], xr half [S*2*H], gu_raw f32 [S*2*I],
// act half [S*NPIECE*I], d_raw f32 [S*H]; hot_y optional [S, H]; out [M, H] (fp32 partial or bf16)
void moe_cold_m(torch::Tensor x, torch::Tensor sel, torch::Tensor wt, int64_t tpk,
                torch::Tensor gu_ptr, torch::Tensor gu_k2, torch::Tensor dn_ptr, torch::Tensor dn_meta, int maxn,
                torch::Tensor runs, torch::Tensor xr, torch::Tensor gu_raw, torch::Tensor act, torch::Tensor d_raw,
                c10::optional<torch::Tensor> hot_y, torch::Tensor is_hot, c10::optional<torch::Tensor> res,
                torch::Tensor out, double limit, int64_t I)
{
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int S = (int) sel.numel();
    const int M = S / (int) tpk;
    TORCH_CHECK(S <= MAXS && tpk <= 8 && M * tpk == S);
    TORCH_CHECK(runs.numel() >= 2 + 2 * S + 1 && xr.numel() >= (int64_t) S * 2 * HDIM && gu_raw.numel() >= (int64_t) S * 2 * I
                && act.numel() >= (int64_t) S * NPIECE * I && d_raw.numel() >= (int64_t) S * HDIM, "moe_cold_m: scratch");
    const auto* sp = (const int64_t*) sel.data_ptr();
    const auto* gp = (const int64_t*) gu_ptr.data_ptr();
    const auto* gk = (const int*) gu_k2.data_ptr();
    const auto* dp = (const int64_t*) dn_ptr.data_ptr();
    const auto* dm = (const int*) dn_meta.data_ptr();
    int* rp = (int*) runs.data_ptr();
    half* xp = (half*) xr.data_ptr();
    float* gr = (float*) gu_raw.data_ptr();
    half* ap = (half*) act.data_ptr();
    float* dr = (float*) d_raw.data_ptr();
    moe_runs_kernel<<<1, 64, 0, st>>>(sp, gp, S, rp);
    moe_rot_kernel<<<S * 2, 512, 0, st>>>((const __nv_bfloat16*) x.data_ptr(), sp, gp, xp, (int) tpk);
    if (g_wide_gu) moe_gu_m_kernel<true><<<S * 2 * ((int) I / tile_cols<true>()), THREADS, 0, st>>>(sp, rp, S, xp, gp, gk, gr, (int) I);
    else           moe_gu_m_kernel<false><<<S * 2 * ((int) I / tile_cols<false>()), THREADS, 0, st>>>(sp, rp, S, xp, gp, gk, gr, (int) I);
    moe_act_kernel<<<S * NPIECE, 512, 0, st>>>(sp, gp, dp, dm, gr, ap, (float) limit, (int) I, 1, 0);
    if (g_wide_dn)
    {
        const int ng = CEIL_DIVIDE(maxn, tile_cols<true>());
        moe_down_m_kernel<true><<<S * NPIECE * ng, THREADS, 0, st>>>(sp, rp, S, dp, dm, ap, dr, ng, (int) I);
    }
    else
    {
        const int ng = CEIL_DIVIDE(maxn, tile_cols<false>());
        moe_down_m_kernel<false><<<S * NPIECE * ng, THREADS, 0, st>>>(sp, rp, S, dp, dm, ap, dr, ng, (int) I);
    }
    moe_out_m_kernel<<<dim3(HDIM / 128, M), 256, 0, st>>>(sp, (const float*) wt.data_ptr(), dp, dm, dr,
                                                         hot_y ? (const float*) hot_y->data_ptr() : nullptr,
                                                         (const bool*) is_hot.data_ptr(),
                                                         res ? (const __nv_bfloat16*) res->data_ptr() : nullptr,
                                                         out.data_ptr(), (int) tpk, out.scalar_type() == at::kBFloat16 ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("moe_cold", &moe_cold);
    m.def("set_wide", &set_wide);
    m.def("set_ksplit", &set_ksplit);
    m.def("moe_cold_m", &moe_cold_m);
}

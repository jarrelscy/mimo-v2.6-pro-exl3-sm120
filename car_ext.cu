// Custom one-shot "push" all-reduce over PCIe P2P (CUDA IPC), for tiny decode messages (<= NMAX fp32).
// Each rank pushes its chunk into every peer's receive buffer recv[parity][src], fences (system scope), then
// releases a per-(block, src) flag on the peer; after acquiring all peer flags it sums the W partials in fixed
// rank order (bit-identical result on every rank) in place. Epochs live in device memory (graph-replay safe).
// Double-buffering by epoch parity makes a single flag round per call sufficient: a rank can only overwrite
// recv[p] of epoch e after every peer released epoch e-1, which they do only after finishing their e-2 reads.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuda_bf16.h>

#define MAXW 8

struct CarState {
    int rank = -1, W = 0, nmax = 0, nb = 0, fence = 1;
    float* recv = nullptr;       // local [2][W][nmax]
    uint32_t* flags = nullptr;   // local [nb][W]
    uint32_t* counters = nullptr;  // local [nb]
    float* peer_recv[MAXW];
    uint32_t* peer_flags[MAXW];
};
static CarState S;

struct Ptrs { float* recv[MAXW]; uint32_t* flags[MAXW]; };

__device__ __forceinline__ void st_release_sys(uint32_t* p, uint32_t v)
{
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p)
{
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void prefetch_range(const char* p, size_t bytes, int b, int nb)
{
    // each thread prefetches 128-byte lines, grid-strided over [p, p + bytes)
    const size_t lines = bytes / 128;
    for (size_t i = (size_t) b * blockDim.x + threadIdx.x; i < lines; i += (size_t) nb * blockDim.x)
        asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p + i * 128));
}

struct Pf { const char* p[3]; unsigned long long b[3]; };
// optional fused epilogue: xo = bf16(x + bf16(sum)); h = bf16(w * bf16(xo * rsqrt(mean(xo^2) + eps)))  (nar must be 1)
struct Nrm { __nv_bfloat16* x; const __nv_bfloat16* w; __nv_bfloat16* h; float eps; int on; };
__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

template <int W>
__global__ void car_kernel(float* __restrict__ data, int n, int rank, int nmax, Ptrs P,
                           float* __restrict__ my_recv, uint32_t* __restrict__ my_flags, uint32_t* __restrict__ counters, int fence, int nar, Pf pf, Nrm nm)
{
    const int tid = threadIdx.x;
    if ((int) blockIdx.x >= nar)
    {
        // extra blocks: L2 prefetch (evict_last) of upcoming weights while the all-reduce waits on PCIe
        const int pb = blockIdx.x - nar, npb = gridDim.x - nar;
        #pragma unroll
        for (int k = 0; k < 3; ++k) if (pf.b[k]) prefetch_range(pf.p[k], pf.b[k], pb, npb);
        return;
    }
    const int b = blockIdx.x, nb = nar;
    __shared__ uint32_t ep_s;
    if (tid == 0) { uint32_t e = counters[b] + 1; counters[b] = e; ep_s = e; }
    __syncthreads();
    const uint32_t ep = ep_s;
    const int par = ep & 1;
    const int chunk = n / nb;
    const int n4 = chunk / 4;
    float4* src = (float4*) (data + (size_t) b * chunk);
    // push
    #pragma unroll
    for (int p = 0; p < W; ++p)
    {
        if (p == rank) continue;
        float4* dst = (float4*) (P.recv[p] + ((size_t) (par * W + rank)) * nmax + (size_t) b * chunk);
        for (int i = tid; i < n4; i += blockDim.x) dst[i] = src[i];
    }
    if (fence) __threadfence_system();
    __syncthreads();
    if (tid < W && tid != rank) { if (!fence) __threadfence_system(); st_release_sys(P.flags[tid] + b * W + rank, ep); }
    if (tid < W && tid != rank) { while (ld_acquire_sys(my_flags + b * W + tid) < ep) {} }
    __syncthreads();
    // reduce in fixed order
    for (int i = tid; i < n4; i += blockDim.x)
    {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        #pragma unroll
        for (int s = 0; s < W; ++s)
        {
            float4 v;
            if (s == rank) v = src[i];
            else v = __ldcg(((const float4*) (my_recv + ((size_t) (par * W + s)) * nmax + (size_t) b * chunk)) + i);
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
        if (nm.on)
        {
            // fused residual add; keep the new x (fp32 of bf16) in data for the norm pass
            const int e = (b * chunk) / 4 + i;
            __nv_bfloat162* xp = (__nv_bfloat162*) nm.x + 2 * e;
            float2 x01 = __bfloat1622float2(xp[0]), x23 = __bfloat1622float2(xp[1]);
            acc.x = bfr(x01.x + bfr(acc.x)); acc.y = bfr(x01.y + bfr(acc.y));
            acc.z = bfr(x23.x + bfr(acc.z)); acc.w = bfr(x23.y + bfr(acc.w));
            xp[0] = __floats2bfloat162_rn(acc.x, acc.y); xp[1] = __floats2bfloat162_rn(acc.z, acc.w);
        }
        src[i] = acc;
    }
    if (nm.on)
    {
        __shared__ float red[32];
        float ss = 0.f;
        for (int i = tid; i < n4; i += blockDim.x) { float4 v = src[i]; ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w; }
        #pragma unroll
        for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if ((tid & 31) == 0) red[tid >> 5] = ss;
        __syncthreads();
        if (tid < 32)
        {
            float t = tid < (int) (blockDim.x >> 5) ? red[tid] : 0.f;
            #pragma unroll
            for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
            if (tid == 0) red[0] = t;
        }
        __syncthreads();
        const float rs = rsqrtf(red[0] / (float) n + nm.eps);
        for (int i = tid; i < n4; i += blockDim.x)
        {
            float4 v = src[i];
            const __nv_bfloat162* wp = (const __nv_bfloat162*) nm.w + 2 * i;
            float2 w01 = __bfloat1622float2(wp[0]), w23 = __bfloat1622float2(wp[1]);
            __nv_bfloat162* hp = (__nv_bfloat162*) nm.h + 2 * i;
            hp[0] = __floats2bfloat162_rn(w01.x * bfr(v.x * rs), w01.y * bfr(v.y * rs));
            hp[1] = __floats2bfloat162_rn(w23.x * bfr(v.z * rs), w23.y * bfr(v.w * rs));
        }
    }
}

__device__ __forceinline__ void load_range(const char* p, size_t bytes, int b, int nb, uint32_t* sink)
{
    // real 16-byte loads kept in L2 with evict_last priority; xor-reduced into a sink so they are not elided
    const size_t n16 = bytes / 16;
    uint32_t acc = 0;
    for (size_t i = (size_t) b * blockDim.x + threadIdx.x; i < n16; i += (size_t) nb * blockDim.x)
    {
        uint32_t a, bb, c, d;
        asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(a), "=r"(bb), "=r"(c), "=r"(d) : "l"(p + i * 16));
        acc ^= a ^ bb ^ c ^ d;
    }
    if (acc == 0x9e3779b9u) sink[0] = acc;  // practically never taken
}

__global__ void prefetch_kernel(const char* p, size_t bytes, int mode, uint32_t* sink)
{
    if (mode == 0) prefetch_range(p, bytes, blockIdx.x, gridDim.x);
    else load_range(p, bytes, blockIdx.x, gridDim.x, sink);
}
static uint32_t* g_sink = nullptr;

void prefetch_l2(torch::Tensor t, int64_t offset, int64_t bytes, int64_t nblocks, int64_t threads, int64_t mode)
{
    if (!g_sink) C10_CUDA_CHECK(cudaMalloc(&g_sink, 64));
    const at::cuda::OptionalCUDAGuard guard(t.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const size_t tot = t.numel() * t.element_size();
    if ((size_t) offset >= tot) return;
    size_t b = std::min((size_t) bytes, tot - (size_t) offset);
    prefetch_kernel<<<nblocks, threads, 0, st>>>((const char*) t.data_ptr() + offset, b, (int) mode, g_sink);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// returns uint8 [2, sizeof(cudaIpcMemHandle_t)] (recv, flags)
torch::Tensor car_init(int64_t rank, int64_t W, int64_t nmax, int64_t nb)
{
    TORCH_CHECK(W <= MAXW);
    S.rank = rank; S.W = W; S.nmax = nmax; S.nb = nb;
    C10_CUDA_CHECK(cudaMalloc(&S.recv, sizeof(float) * 2 * W * nmax));
    C10_CUDA_CHECK(cudaMalloc(&S.flags, sizeof(uint32_t) * nb * W));
    C10_CUDA_CHECK(cudaMalloc(&S.counters, sizeof(uint32_t) * nb));
    C10_CUDA_CHECK(cudaMemset(S.recv, 0, sizeof(float) * 2 * W * nmax));
    C10_CUDA_CHECK(cudaMemset(S.flags, 0, sizeof(uint32_t) * nb * W));
    C10_CUDA_CHECK(cudaMemset(S.counters, 0, sizeof(uint32_t) * nb));
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    auto h = torch::empty({2, (int64_t) sizeof(cudaIpcMemHandle_t)}, torch::kUInt8);
    cudaIpcMemHandle_t hr, hf;
    C10_CUDA_CHECK(cudaIpcGetMemHandle(&hr, S.recv));
    C10_CUDA_CHECK(cudaIpcGetMemHandle(&hf, S.flags));
    memcpy(h.data_ptr<uint8_t>(), &hr, sizeof(hr));
    memcpy(h.data_ptr<uint8_t>() + sizeof(hr), &hf, sizeof(hf));
    return h;
}

// all: uint8 [W, 2, sizeof(handle)] (CPU)
void car_set_fence(int64_t f) { S.fence = (int) f; }

void car_open(torch::Tensor all)
{
    for (int p = 0; p < S.W; ++p)
    {
        if (p == S.rank) { S.peer_recv[p] = S.recv; S.peer_flags[p] = S.flags; continue; }
        cudaIpcMemHandle_t hr, hf;
        const uint8_t* base = all.data_ptr<uint8_t>() + (size_t) p * 2 * sizeof(hr);
        memcpy(&hr, base, sizeof(hr)); memcpy(&hf, base + sizeof(hr), sizeof(hf));
        void* pr; void* pf;
        C10_CUDA_CHECK(cudaIpcOpenMemHandle(&pr, hr, cudaIpcMemLazyEnablePeerAccess));
        C10_CUDA_CHECK(cudaIpcOpenMemHandle(&pf, hf, cudaIpcMemLazyEnablePeerAccess));
        S.peer_recv[p] = (float*) pr; S.peer_flags[p] = (uint32_t*) pf;
    }
}

static Pf make_pf(const std::vector<torch::Tensor>& ts, const std::vector<int64_t>& bytes)
{
    Pf pf; for (int k = 0; k < 3; ++k) { pf.p[k] = nullptr; pf.b[k] = 0; }
    for (size_t k = 0; k < ts.size() && k < 3; ++k)
    {
        const size_t tot = ts[k].numel() * ts[k].element_size();
        pf.p[k] = (const char*) ts[k].data_ptr();
        pf.b[k] = std::min((size_t) bytes[k], tot);
    }
    return pf;
}

void car_allreduce(torch::Tensor x, int64_t threads, int64_t nb, std::vector<torch::Tensor> pft, std::vector<int64_t> pfb, int64_t npf,
                   c10::optional<torch::Tensor> nx, c10::optional<torch::Tensor> nw, c10::optional<torch::Tensor> nh, double eps)
{
    Nrm nm; nm.on = 0; nm.x = nullptr; nm.w = nullptr; nm.h = nullptr; nm.eps = (float) eps;
    if (nx.has_value())
    {
        TORCH_CHECK(nw.has_value() && nh.has_value());
        TORCH_CHECK(nx->scalar_type() == at::kBFloat16 && nw->scalar_type() == at::kBFloat16 && nh->scalar_type() == at::kBFloat16);
        TORCH_CHECK(nx->numel() == x.numel() && nw->numel() == x.numel() && nh->numel() == x.numel());
        nm.on = 1; nm.x = (__nv_bfloat16*) nx->data_ptr(); nm.w = (const __nv_bfloat16*) nw->data_ptr(); nm.h = (__nv_bfloat16*) nh->data_ptr();
    }
    Pf pf = make_pf(pft, pfb);
    if (pft.empty()) npf = 0;
    if (nb <= 0) nb = S.nb;
    TORCH_CHECK(!nm.on || nb == 1, "car: fused norm needs nb == 1");
    TORCH_CHECK(nb <= S.nb);
    TORCH_CHECK(x.scalar_type() == at::kFloat && x.is_contiguous());
    const int n = (int) x.numel();
    TORCH_CHECK(n <= S.nmax && n % (4 * nb) == 0, "car: bad size");
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    Ptrs P;
    for (int p = 0; p < S.W; ++p) { P.recv[p] = S.peer_recv[p]; P.flags[p] = S.peer_flags[p]; }
    switch (S.W)
    {
        case 2: car_kernel<2><<<(int) (nb + npf), threads, 0, st>>>(x.data_ptr<float>(), n, S.rank, S.nmax, P, S.recv, S.flags, S.counters, S.fence, (int) nb, pf, nm); break;
        case 4: car_kernel<4><<<(int) (nb + npf), threads, 0, st>>>(x.data_ptr<float>(), n, S.rank, S.nmax, P, S.recv, S.flags, S.counters, S.fence, (int) nb, pf, nm); break;
        case 8: car_kernel<8><<<(int) (nb + npf), threads, 0, st>>>(x.data_ptr<float>(), n, S.rank, S.nmax, P, S.recv, S.flags, S.counters, S.fence, (int) nb, pf, nm); break;
        default: TORCH_CHECK(false, "car: W must be 2/4/8");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("init", &car_init);
    m.def("open", &car_open);
    m.def("set_fence", &car_set_fence);
    m.def("prefetch_l2", &prefetch_l2);
    m.def("allreduce", &car_allreduce, py::arg("x"), py::arg("threads") = 256, py::arg("nb") = -1,
          py::arg("pft") = std::vector<torch::Tensor>(), py::arg("pfb") = std::vector<int64_t>(), py::arg("npf") = 0,
          py::arg("nx") = py::none(), py::arg("nw") = py::none(), py::arg("nh") = py::none(), py::arg("eps") = 1e-6);
}

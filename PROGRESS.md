# MiMo-V2.6-Pro-EXL3 fast decode — progress log

Box: 4x RTX PRO 6000 Blackwell (SM120, 96 GB), PCIe (NODE topology, P2P OK), no NVLink.
Reference: /data/Jarrel/mimo-pro-exl3-smoke (untouched). Work dir: /data/Jarrel/mimo-pro-exl3-fast.

## End-to-end table (single stream, greedy)

| # | change | ms/step (short) | tok/s short | tok/s @2K | tok/s @4K | coherence |
|---|--------|-----------------|-------------|-----------|-----------|-----------|
| 0 | reference loader (PP over 4 GPUs, eager, per-expert python loop) | ~345 | 2.85-2.92 | - | - | Paris / chat stops / story OK |
| 1 | TP4 (mimo_tp.py): all fused kernels, 1 CUDA graph/step, NCCL all-reduce (P2P_LEVEL=SYS) | 12.64 | 79.1 | 76.5 | 75.3 | chat identical to ref; story same opening, coherent; completion drifts after 6 tokens (numeric); needle 2K/4K PASS |
| 2 | + custom P2P one-shot all-reduce (car_ext.cu, nb=1) + fp8 qkv GEMV v2 (BK=512) + Triton gate GEMV | 11.80 | 83.0 | 81.6 | 77.2 | chat stops at <\|im_end\|>, story coherent 300 tok, needle 2K/4K PASS |

## Layer-level (single GPU3, layer 60 = SWA + MoE with 73 hot NVFP4 experts)

| change | layer decode ms | kernels/layer | accuracy |
|--------|-----------------|---------------|----------|
| reference layer (eager) | 5.23 | ~hundreds (python loop over 8 experts, 3 EXL3 gemms each + hadamards) | - |
| fp8 Triton qkv GEMV + static KV + Triton split-KV attn w/ sinks + exl3_mgemm grouped cold experts + Triton NVFP4 hot + CUDA graph | 1.70 (MoE alone 1.70 -> it was the bottleneck) | ~100 | relerr vs ref 1.1e-2 |
| fused cold MoE ext (moe_ext.cu: heterogeneous-K gu/act/down/out kernels reusing exllamav3 gemv_tile) | 0.53 eager / 0.517 graph (MoE 0.213) | 60 | same |
| fused small ops: add+rmsnorm, rope+KV-write (qkv_post), device routing, hot silu*mul folded into nvfp4 down, residual add folded into moe_out | 0.465 eager / 0.459 graph (MoE 0.221, attn 0.30) | 17 | graph==eager bit-exact; dense-fp32 check: ours 2.7-3.6e-3 vs reference 0.8-1.0e-2 |
| TP4 shard (rank-0 slice of the same layer, 4-rank sim summed by hand) | 0.10 graph (L2-warm) | 17 | tp-vs-pp 2.5e-3, tp-vs-ref 1.0e-2 (= pp-vs-ref) |

Accuracy note: relerr "vs reference" is dominated by the reference's own error: against a dense fp32 reconstruction
of the same quantized weights our fused path is 2.7-3.6e-3, the reference 0.8-1.0e-2 (its int8-ish internal
gemv path is noisier). So small greedy-text drift vs the reference is expected and is not a quality loss.

## Log
- 2026-09-29 00:25 **TP4** (mimo_tp.py, torchrun 4 procs, NCCL_P2P_LEVEL=SYS, one CUDA graph per rank per step,
  self-feeding token/pos): load 112 s, 71.8 GiB/rank. completion 78.2 tok/s (12.74 ms), chat 77.4, story 79.1
  (12.64 ms median), ctx2000 76.5 (13.11 ms), ctx4000 75.3 (13.25 ms). Needle 2K+4K PASS. Chat identical to the
  reference and stops at <|im_end|>; story identical opening paragraphs to the reference, coherent 300 tokens, no loops;
  completion " Paris. In the early Middle Ages Paris was one of the major cities of the Carolingian Empire..." (reference:
  " Paris. In the city of Paris, there are 12 million people...": same first tokens, drift after "In the" = numeric).
  Prefill is token-by-token through the decode graph: ~77 tok/s (4K prompt = 51 s).
- NCCL all-reduce 24 KB fp32 x4 GPUs: default 18.5 us eager / 32 us in graph; NCCL_P2P_LEVEL=SYS 11.9 eager / 25.8 in
  graph; NCCL_PROTO=LL no change. 140 all-reduces/step -> ~3.6 ms/step of 12.7.
- PP fused full model (smoke_fast.py) currently hits an illegal memory access (single-layer test passes); TP is the product.
- 2026-09-29 00:50 **v2** car + fp8v2 + gate kernel: story 11.80 ms median (83.0 tok/s), completion 12.0 ms, chat 81.1, ctx2000
  81.6, ctx4000 77.2 tok/s; needles PASS; outputs coherent. Load 176 s (page cache cold-ish after nestquant).
- custom all-reduce (car): push to peers' IPC buffers + release/acquire sys-scope flags, epoch-parity double buffer,
  fixed-order sum => bit-identical on all ranks and bit-exact vs fixed-order reference. nb=1/128 thr best:
  ~9.5 us alone, ~8.5 us net in a graph interleaved with kernels vs NCCL ~13 us net (NCCL's 26 us "in graph" number
  is only for back-to-back chained all-reduces). More blocks/threads are SLOWER (8 blocks: 23 us). Dropping the
  per-thread __threadfence_system (release cumulativity) makes no difference.
- fp8 qkv GEMV: BK=128 scale-per-iteration kernel 38.9 us L2-cold -> 2D-acc BK=512 kernel 28.4 us (1.47 TB/s).
- gate logits: torch.mm(h.float(), W.T) 10.3 us cold -> Triton gate_gemv 8.3 us (also drops the h.float() kernel).
- OpenAI server (server_tp.py / run_server.sh, port 8003) works: /v1/models (local, mimo-v2.6-pro-exl3),
  /v1/completions, /v1/chat/completions (reasoning_content split, tool_calls parsing, SSE streaming + usage,
  stop strings, seeded Gumbel-max sampling in-graph, exact-extension KV prefix reuse across turns).
- MoE shard L2-cold microbench (mb_moe.py: graph of 16 different tokens): layer 60 (hot) 99.4 us, of which the
  3 NVFP4 hot GEMVs were 49 us. Hot gate/up BN=16/BK=256 -> BN=2/BK=1024: 99.4 -> 69.8 us/layer (27 hot layers).
  Split-K for the cold EXL3 gu/down (moe_ext set_ksplit, deterministic partial sums) does NOT help (98-148 us);
  cold layer: gu 23 us (12.6 MB of 2-bit trellis, 0.55 TB/s) + down 16 us: trellis decode is ALU-bound at bs=1.
- L2 prefetch microbench (mb_pf.py, GPU3, 24 MB unrelated traffic between prefetch and GEMV): `prefetch.global.L2::evict_last`
  of the first 48 MB of the bf16 o_proj shard (32 blocks) -> GEMV 34.9 -> 22.5 us; 16 MB of the fp8 qkv -> 30.8 -> ~27 us.
  Plain loads (ld.L1::no_allocate, normal L2 priority) do NOT survive the 24 MB of traffic (no gain) -> evict_last is what
  matters. The prefetch kernel itself costs 5-15 us, so it only pays if hidden: implemented as extra blocks of the car
  all-reduce kernel (the AR spends ~8 us waiting on PCIe with DRAM idle), and the AR before the ALU-bound cold MoE
  (MIMO_PF="A=q42" = prefetch the next layer's qkv while the trellis MoE runs). E2E A/B pending (ab_tp.py).
- Fused AR epilogue (MIMO_FUSENORM=1): x = bf16(x + bf16(sum)), h = rmsnorm(x)*w inside the car kernel (removes 140
  add_rmsnorm launches/step). Test: car_norm_test.py. E2E A/B pending.
- 2026-09-29 **MTP self-speculation (spec_tp.py, kernels_m.py, moe_ext moe_cold_m)**: the checkpoint's MTP layer
  (model.mtp.layers.0: eh_proj(cat(enorm(emb), hnorm(h))) + SWA block + final norm) drafts K tokens; one CUDA graph
  verifies MM=K+1 rows [x0, d1..dK], accepts n = prefix match, emits g[0..n], then drafts the next K tokens (batched MTP
  over the verified rows + a K-1 single-row MTP chain). Everything (accept count, position advance, slot math) is on
  device, so a step is one graph replay plus one 9-int readback. SWA KV is a ring of R=128+8 slots so rejected rows
  never corrupt the window; rejected full-attn rows are overwritten by the next step. Prefill is chunked through the
  same graph with forced acceptance (MM tokens/step). Temperature>0 uses a separate MM=1 graph (Gumbel, same KV).
  - M-row kernels: the tensor-core (tl.dot, x padded to 16 rows) fp8 qkv GEMV is flat in M: 29.6 us at M=1/4/8 (vs
    29.1 for the M=1 kernel); bf16 o_proj ~32.5 us flat. A 3D-accumulator (CUDA-core) variant grows with M (43 us at M=4).
    Routing, add+rmsnorm, rope/KV-write: bit-exact vs the M=1 kernels. Attention: full layers bit-exact, SWA+sink
    4-9e-3 maxdiff vs torch.
  - moe_cold_m: exllamav3's gemv_tile natively handles up to 8 rows of the same expert with MMA, so rows sharing an expert
    are ~free and cost scales with UNIQUE experts: L10 shard M=1 75 us, M=3 (19 unique) 110, M=4 (19 unique) 114, M=8 247
    (L2-warm, GPU shared). Bit-exact vs looping the M=1 path (test_moe_m.py).
  - Logic test (test_spec_w1.py, W=1, 1 layer, GPU3): K=1/3/7 greedy output identical to K=0; with an oracle drafter every
    step accepts K; with a corrupted oracle acceptance stops at the corruption; prefix reuse (cached=66) identical to fresh.
  - Server: MIMO_SPEC=K ./run_server.sh routes generate through Spec (greedy -> K+1 verify graph, sampled -> MM=1 graph).
  - Draft weights in fp8 (MIMO_DRAFT_FP8=1, default): the draft-only lm_head (470 MB/rank bf16) and the MTP dense MLP
    (checkpoint is fp8 already; was dequantized to bf16) use 128x128-block fp8 + the tensor-core fp8 GEMV. Output tokens are
    unaffected by construction (drafts are always verified by the bf16 main model); only acceptance can move. lm_head
    1-row: 293 -> 155 us (GPU3, shared).
  - Wide prefill graph (MIMO_SPEC_PF=8): prompt chunks go through an 8-row prefill-only graph (no draft chain); the last
    <= K+1 prompt tokens go through the decode graph, so drafts/sampling are set up exactly as before. W=1 test with a
    301-token prompt (crosses the 128 SWA window): identical to K=0.
  - Multi-row all-reduce knob MIMO_SPEC_CAR_NB (needs MIMO_CAR_NB >= it); bench car_bench_m.py.

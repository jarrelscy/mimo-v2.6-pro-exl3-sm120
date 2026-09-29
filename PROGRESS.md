# MiMo-V2.6-Pro-EXL3 fast decode — progress log

Box: 4x RTX PRO 6000 Blackwell (SM120, 96 GB), PCIe (NODE topology, P2P OK), no NVLink.
Reference: /data/Jarrel/mimo-pro-exl3-smoke (untouched). Work dir: /data/Jarrel/mimo-pro-exl3-fast.

## End-to-end table (single stream, greedy)

| # | change | ms/step (short) | tok/s short | tok/s @2K | tok/s @4K | coherence |
|---|--------|-----------------|-------------|-----------|-----------|-----------|
| 0 | reference loader (PP over 4 GPUs, eager, per-expert python loop) | ~345 | 2.85-2.92 | - | - | Paris / chat stops / story OK |
| 1 | TP4 (mimo_tp.py): all fused kernels, 1 CUDA graph/step, NCCL all-reduce (P2P_LEVEL=SYS) | 12.64 | 79.1 | 76.5 | 75.3 | chat identical to ref; story same opening, coherent; completion drifts after 6 tokens (numeric); needle 2K/4K PASS |
| 2 | + custom P2P one-shot all-reduce (car_ext.cu, nb=1) + fp8 qkv GEMV v2 (BK=512) + Triton gate GEMV | 11.80 | 83.0 | 81.6 | 77.2 | chat stops at <\|im_end\|>, story coherent 300 tok, needle 2K/4K PASS |
| 3 | + L2 prefetch (evict_last) of next qkv/o_proj by extra AR blocks (MIMO_PF=A=q42/D=o24) + car 256 threads | 10.67 | 93.6 (code 91.4) | 90.1 | 88.6 | bit-identical to row 2 tokens; needles PASS |
| 4 | + MTP self-speculation K=2 (spec_tp.py; fp8 drafts, 512-thread multi-row car) = server default | 18.8/step, 2.02-2.49 tok/step | 107.4 (code 127.4, chat 119.0) | 116.2 | 114.5 | chat identical to base; story/code coherent, drift vs base after 80/219 tok; needles PASS; server suite PASS |

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
- 2026-09-29 14:24 **batch2 (first full-model spec run, car nb=1 / 128 threads)**. Decode tok/s (greedy, 300-tok story/code, 2K/4K needle):

  | run | completion | chat | story | code | ctx2K | ctx4K | ms/step | tok/step (story/code) |
  |---|---|---|---|---|---|---|---|---|
  | base (M=1 graph) | 88.0 | 86.8 | 87.3 | 85.4 | 84.3 | 83.3 | 11.4 | 1 |
  | spec K=0 (M-row kernels, MM=1) | 79.3 | 77.6 | 78.7 | 77.4 | 78.1 | 76.1 | ~12.7 | 1 |
  | spec K=1 | 110.5 | 107.8 | 101.4 | 109.2 | 100.7 | 100.7 | 16.4-18.0 | 1.72 / 1.90 |
  | spec K=2 | 95.9 | 107.7 | 96.9 | 114.3 | 102.4 | 102.0 | 20.2-22.1 | 2.02 / 2.49 |
  | spec K=3 | 85.9 | 91.5 | 82.9 | 102.0 | 87.4 | 114.7* | 24.1-26.2 | 2.03 / 2.65 |
  | spec K=3, bf16 drafts | 84.2 | 86.7 | 78.8 | 99.2 | 84.3 | 111.9* | 24.5-26.9 | 2.03 / 2.67 |
  | spec K=4 | 76.1 | 78.4 | 71.6 | 89.9 | 76.2 | 100.5* | 27.2-30.0 | 2.03 / 2.67 |

  (*ctx4K is only 10 new tokens / 3 steps: noisy.) Needles 2K/4K PASS for every K; chat identical to base for K>=2;
  story/code diverge from base after 12-219 tokens (M-row kernels round differently, temp-0 drift; text coherent).
  MTP acceptance: first draft ~72-90%, but acceptance of draft 2+ is low (story K=3 hist [36,75,31,5]).
  fp8 drafts: same acceptance as bf16 drafts (hist within noise) and ~1-1.3 ms/step faster -> kept.
  Profile K=3 step (23.9 ms kernel time): **all-reduce 7.97 ms (33%)**, cold MoE 6.45, fp8 GEMV 2.91, bf16 GEMV 2.82,
  hot MoE 1.48. Base M=1 step (11.64 ms): cold MoE 2.99, bf16 GEMV 2.59, fp8 GEMV 1.95, all-reduce 1.94, gate+route 0.78.
  => the multi-row all-reduce was the problem: car_bench_m: 4 rows = 51 us at nb=1/128 thr vs 13 us for 1 row.
- Prefetch A/B (ab_pf, base path): pf=A=q42/D=o24 10.92 ms (91.5 tok/s) vs 11.42 (87.5), bit-identical -> DEFAULT ON.
  fuse=1 (norm fused into the AR epilogue) 12.24 ms = SLOWER and not bit-identical to the unfused run -> not shipped
  (the single 128-thread block doing the norm serializes behind the PCIe wait).
- car thread sweep (car_bench_m2.py, all configs bit-exact): the one-shot push is limited by outstanding remote stores per
  block, not by block count: 4 rows 51 -> 22.1 us with 512 threads (nb=1); 2 rows 27.6 -> 18.6; 1 row 15.0 -> 11.5 us
  with 256 threads. More blocks are worse (each block adds a flag handshake). car now picks 256 (1 row) / 512 (multi-row).
- 2026-09-29 14:35 **batch3** (car auto threads + prefetch default + prefetch inside the spec step). Base 10.67-10.76 ms
  (93.7 tok/s) with thr1=256 vs 10.80-10.89 with 128, both bit-identical. Spec decode tok/s:

  | run | completion | chat | story | code | ctx2K | ctx4K | ms/step (story) | tok/step story/code |
  |---|---|---|---|---|---|---|---|---|
  | base | 94.9 | 92.9 | 93.6 | 91.4 | 90.1 | 88.6 | 10.67 | 1 |
  | K=1 | 115.2 | 112.7 | 106.3 | 113.3 | 107.6 | 107.5 | 16.17 | 1.72 / 1.90 |
  | **K=2** | 105.3 | **119.0** | **107.4** | **127.4** | **116.2** | 114.5 | 18.80 | 2.02 / 2.49 |
  | K=3 | 98.9 | 103.9 | 94.6 | 115.8 | 96.7 | 126.8* | 21.50 | 2.03 / 2.65 |

  K=3 step 23.9 -> 21.5 ms from the all-reduce fix alone. K=2 chosen (best or tied on every realistic test).
- Server (MIMO_SPEC=2 ./run_server.sh, test_server.py): completion " Paris. In the early Middle Ages...", chat stops,
  thinking split (17*23 -> 391), SSE stream, stop strings, sampled t=0.8, tool call, turn-2 prefix reuse (cached 15): all PASS.

## Profile before / after (kernel time per decode step, rank 0, prof_spec.py / smoke_tp.py --profile)

| bucket | v1 (NCCL AR) 1 tok | v2+ base 1 tok | spec K=3, 4 rows (nb=1/128 thr car) |
|---|---|---|---|
| all-reduce | 27.9 ms under profiler (NCCL LL, ~13 us net each) | 1.94 | 7.97 -> ~3.1 after the 512-thread fix (22 us x 142) |
| cold MoE (EXL3 trellis, ALU-bound) | 2.8 | 2.99 | 6.45 |
| bf16 GEMV (o_proj, lm_head, eh_proj) | 2.6 | 2.59 | 2.82 |
| fp8 GEMV (qkv, drafts) | 2.5 | 1.95 | 2.91 |
| hot MoE (NVFP4) | 1.1 | 0.49 | 1.48 |
| gate + route | 0.9 | 0.78 | 0.87 |
| attention | 0.2 | 0.31 | 0.67 |
| norm/rope/kv | 0.5 | 0.47 | 0.53 |
| total | ~12.6 wall | 11.64 (10.67 wall with prefetch) | 23.93 |

(v1 NCCL time is inflated by the profiler; the E2E difference v1->v2 was 0.84 ms/step.)

## What did not work (and why)
- Split-K for the cold EXL3 MoE (moe_ext set_ksplit): 98-148 us vs 99 us/layer. The 2-bit trellis decode is ALU-bound at
  bs=1, not bandwidth-bound, so more CTAs over K only add a reduction.
- More all-reduce blocks (nb 2-8): every block runs its own sys-scope flag handshake over PCIe; 1 row nb=8 = 30 us vs
  12. Multi-row wants more threads in ONE block (outstanding remote stores), not more blocks.
- Fused AR epilogue (residual add + rmsnorm inside car, MIMO_FUSENORM=1): 12.24 vs 11.42 ms, and not bit-identical to
  the unfused run. A single block doing the norm serializes behind the PCIe wait; kept as an option, off.
- Plain-load L2 warmup (ld.L1::no_allocate) instead of prefetch.global.L2::evict_last: the lines are evicted by the
  24+ MB of intervening traffic, no gain. evict_last is what makes the prefetch survive.
- CUDA-core M-row GEMVs (3D accumulator): cost grows with M (43 us at M=4 vs 29.6 flat for tl.dot tensor-core ones).
- Deeper speculation (K>=3): MTP acceptance of draft 2+ is low (story K=3 accept hist [36,75,31,5]); each extra row
  costs ~2.7 ms (mostly more unique cold experts to decode), so K=3/4 lose except on very predictable text.
- int8 dp4a mul1 GEMV (upstream exl3_gemv_int8): not ported. Dense-only cooperative kernel, slightly lossy, and its cost is
  linear in rows while our MMA gemv_tile makes extra rows of the same expert nearly free; at best ~3% at M=1.
- PP (pipeline over 4 GPUs) fused full model: illegal memory access at full depth; TP4 is strictly better for bs=1.

## Remaining bottleneck
- Plain decode (10.67 ms): cold EXL3 MoE ~3.0 ms (trellis decode ALU-bound; floor set by the codebook math, not DRAM),
  bf16/fp8 GEMVs ~4.5 ms (near DRAM bandwidth for 1.2 GB/rank/step), all-reduce ~1.9 ms for 142 ARs at ~11.5 us each
  (PCIe latency floor; only fewer ARs would help).
- Spec K=2 (18.8 ms for 3 rows): cold MoE scales with UNIQUE experts (3 rows ~ 19-22 of 384 vs 8), so verification of
  each extra row costs ~2.5-3 ms; the MTP draft passes (batched + chain, 3 ARs each) ~1.5 ms. Better drafts (higher
  acceptance of draft 2+, e.g. a DFlash-style block drafter) are the lever with the most headroom; faster cold-expert
  decode helps both paths.

## Reproduce
```bash
cd /data/Jarrel/mimo-pro-exl3-fast && source env.sh
export NCCL_P2P_LEVEL=SYS OMP_NUM_THREADS=8
# kernel/logic tests (1 GPU)
CUDA_VISIBLE_DEVICES=3 python test_kernels_m.py; CUDA_VISIBLE_DEVICES=3 python test_moe_m.py 10
CUDA_VISIBLE_DEVICES=3 python test_spec_w1.py
# full model (4 GPUs, ~72 GiB each): base + spec smoke/speed, A/B of knobs, profile, all-reduce sweeps
python -m torch.distributed.run --nproc-per-node 4 smoke_tp.py                     # base path, smoke + tok/s
python -m torch.distributed.run --nproc-per-node 4 spec_smoke.py --ks 0,1,2,3      # base + spec K, smoke + tok/s
python -m torch.distributed.run --nproc-per-node 4 ab_tp.py --configs "pf=;thr1=128|pf=A=q42/D=o24;thr1=256" --reps 2
python -m torch.distributed.run --nproc-per-node 4 prof_spec.py --k 2
python -m torch.distributed.run --nproc-per-node 4 car_bench_m2.py
# server (port 8003, spec K=2 by default; MIMO_SPEC=0 for plain decode)
./run_server.sh & python test_server.py
```

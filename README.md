# MiMo-V2.6-Pro EXL3 on 4x RTX PRO 6000 (SM120)

A fast TP4 decode path for the MiMo-V2.6-Pro EXL3 quant on 4x RTX PRO 6000 Blackwell (96 GB, PCIe, no NVLink). It uses
exllamav3 kernels plus custom fused kernels, CUDA graphs and a custom P2P all-reduce.

| stage | ms/step | tok/s |
|---|---|---|
| reference loader | ~345 | 2.9 |
| TP4, fused kernels, one CUDA graph per step | 12.64 | 79.1 |
| + custom all-reduce (car), qkv/gate kernels | 11.80 | 83.0 |

MTP speculative decoding (spec_tp.py) passes single-GPU tests, and the full-model numbers are pending. See PROGRESS.md
for the full log, microbenchmarks and what didn't work.

- `mimo_tp.py`, `kernels*.py`, `moe_ext.cu/.py`: model, fused kernels, EXL3 MoE extension
- `car.py`, `car_ext.cu`: custom all-reduce for PCIe P2P
- `spec_tp.py`: MTP speculative decoding
- `server_tp.py`, `run_server.sh`: OpenAI-compatible server (port 8003)
- `mb_*.py`, `*_bench*.py`, `prof_*.py`: microbenchmarks and profiling; `test_*.py`, `*smoke*.py`: tests
- `env.sh`: environment (CUDA 12.8, TORCH_CUDA_ARCH_LIST=12.0a). The paths are for our box.

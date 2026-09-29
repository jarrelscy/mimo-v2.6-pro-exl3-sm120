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

## Serving

Requirements: 4 GPUs with SM120 (RTX PRO 6000 Blackwell, 96 GB) and ~72 GiB free on each, CUDA 12.8 toolkit (nvcc; the
EXL3 MoE and all-reduce extensions are JIT-built on first start), Python 3.12, and ~285 GB of disk for the weights.

```bash
# 1. weights (backbone + NVFP4 hot experts + EXL3 cold experts; the loader reads layers/*/experts.tar in place)
hf download jarrelscy/MiMo-V2.6-Pro-EXL3 --local-dir /path/to/MiMo-V2.6-Pro-EXL3

# 2. code and environment
git clone https://github.com/jarrelscy/mimo-v2.6-pro-exl3-sm120 && cd mimo-v2.6-pro-exl3-sm120
python3.12 -m venv venv && venv/bin/pip install -r requirements.txt   # torch cu128 wheel
export MIMO_VENV=$PWD/venv CUDA_HOME=/usr/local/cuda-12.8 MIMO_EXL3_DIR=/path/to/MiMo-V2.6-Pro-EXL3

# 3. start the OpenAI-compatible server (TP4, port 8003)
./run_server.sh

curl localhost:8003/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "local", "messages": [{"role": "user", "content": "The capital of France?"}], "max_tokens": 256}'
```

- Endpoints: `/v1/models`, `/v1/completions`, `/v1/chat/completions` (streaming, `reasoning_content`, tool calls),
  `/health`. Any model name is accepted (`local`, `mimo-v2.6-pro-exl3`).
- Single stream: requests are served one at a time. Context is `MIMO_LMAX` tokens (default 32768).
- First start builds the extensions and autotunes the kernels, which takes a few minutes; later starts use the caches.
- `PORT`, `MASTER_PORT` and `MIMO_LMAX` set the port, the torch.distributed port and the context length.
  `MIMO_SPEC=K` turns on MTP speculative decoding (experimental, off by default).
- The GPUs talk over PCIe P2P (`NCCL_P2P_LEVEL=SYS`); the custom all-reduce needs P2P between all four GPUs.

## Files

- `mimo_exl3.py`: reference loader (weight formats, config, backbone reader) that the fast path builds on
- `mimo_tp.py`, `kernels*.py`, `moe_ext.cu/.py`: model, fused kernels, EXL3 MoE extension
- `car.py`, `car_ext.cu`: custom all-reduce for PCIe P2P
- `spec_tp.py`: MTP speculative decoding
- `server_tp.py`, `run_server.sh`: OpenAI-compatible server (port 8003)
- `mb_*.py`, `*_bench*.py`, `prof_*.py`: microbenchmarks and profiling; `test_*.py`, `*smoke*.py`: tests
- `env.sh`: environment (CUDA 12.8, TORCH_CUDA_ARCH_LIST=12.0a); the defaults are our paths, override them as above

#!/usr/bin/env bash
# OpenAI-compatible MiMo-V2.6-Pro-EXL3 server on :8003 (TP4, all 4 GPUs, ~72 GiB/GPU). Hold the box lease while it runs.
cd "$(dirname "$0")" && source env.sh
export NCCL_P2P_LEVEL=SYS MIMO_SAMPLING=1 OMP_NUM_THREADS=8 MIMO_SPEC=${MIMO_SPEC:-2}
exec python -m torch.distributed.run --nproc-per-node 4 --master-port ${MASTER_PORT:-29533} server_tp.py --port ${PORT:-8003} "$@"

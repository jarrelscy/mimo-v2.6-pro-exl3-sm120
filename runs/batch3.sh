#!/bin/bash
# car thread auto-size: base A/B (thr1 128 vs 256, with default prefetch) + spec K=1,2,3 smoke
cd /data/Jarrel/mimo-pro-exl3-fast; source env.sh
trap '/data/Jarrel/coord/boxlease.sh release mimo' EXIT
export NCCL_P2P_LEVEL=SYS OMP_NUM_THREADS=8
C="pf=A=q42/D=o24;thr1=128|pf=A=q42/D=o24;thr1=256|pf=;thr1=128"
timeout 1200 python -m torch.distributed.run --nproc-per-node 4 --master-port 29542 ab_tp.py --configs "$C" --reps 2 --out runs/ab_thr.json > runs/ab_thr.log 2>&1
grep "median" runs/ab_thr.log
timeout 2400 python -m torch.distributed.run --nproc-per-node 4 --master-port 29544 spec_smoke.py --ks 1,2,3 --out runs/spec_smoke2.json > runs/spec_smoke2.log 2>&1
grep -E "\[spec|\[base|---|Error" runs/spec_smoke2.log | tail -40

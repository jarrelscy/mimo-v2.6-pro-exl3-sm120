#!/bin/bash
# one lease window: car tests + in-process A/B; releases the lease on exit
cd /data/Jarrel/mimo-pro-exl3-fast; source env.sh
trap '/data/Jarrel/coord/boxlease.sh release mimo' EXIT
export NCCL_P2P_LEVEL=SYS OMP_NUM_THREADS=8
timeout 300 python -m torch.distributed.run --nproc-per-node 4 --master-port 29541 car_norm_test.py > runs/car_norm_test.log 2>&1
cat runs/car_norm_test.log | grep -v Warning | tail -5
C="pf=|fuse=0|pf=D=q16|pf=D=q42|pf=A=q42|pf=A=q42/D=o24|pf=A=q42/D=o50|pf=A=q42+g9/D=o24|fuse=1|pf=A=q42/D=o24;fuse=1|hot=16,256,4"
timeout 1500 python -m torch.distributed.run --nproc-per-node 4 --master-port 29542 ab_tp.py --configs "$C" --reps 2 --out runs/ab_pf.json > runs/ab_pf.log 2>&1
grep "median" runs/ab_pf.log
timeout 1500 python -m torch.distributed.run --nproc-per-node 4 --master-port 29543 mtp_probe.py --n 200 --k 3 > runs/mtp_probe.log 2>&1
grep -E "steps|Error|error" runs/mtp_probe.log | tail -8

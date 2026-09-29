#!/bin/bash
# one lease window: car tests + spec smoke + prefetch/fuse A/B; releases the lease on exit
cd /data/Jarrel/mimo-pro-exl3-fast; source env.sh
trap '/data/Jarrel/coord/boxlease.sh release mimo' EXIT
export NCCL_P2P_LEVEL=SYS OMP_NUM_THREADS=8
MIMO_CAR_NB=8 timeout 300 python -m torch.distributed.run --nproc-per-node 4 --master-port 29540 car_bench_m.py > runs/car_bench_m.log 2>&1
grep -E "rows=|bit-exact" runs/car_bench_m.log
timeout 300 python -m torch.distributed.run --nproc-per-node 4 --master-port 29541 car_norm_test.py > runs/car_norm_test.log 2>&1
grep -v Warning runs/car_norm_test.log | tail -8
timeout 2400 python -m torch.distributed.run --nproc-per-node 4 --master-port 29544 spec_smoke.py --ks 0,1,2,3,3b,4 --out runs/spec_smoke.json > runs/spec_smoke.log 2>&1
grep -E "\[spec|\[base|---|Error|error" runs/spec_smoke.log | tail -40
timeout 600 python -m torch.distributed.run --nproc-per-node 4 --master-port 29545 prof_spec.py --k 3 --out runs/prof_spec.txt > runs/prof_spec.log 2>&1
grep -E "==|  [a-z]" runs/prof_spec.log | head -30
C="pf=|fuse=0|pf=A=q42|pf=A=q42/D=o24|pf=A=q42+g9/D=o24|fuse=1|pf=A=q42/D=o24;fuse=1|hot=16,256,4"
timeout 1500 python -m torch.distributed.run --nproc-per-node 4 --master-port 29542 ab_tp.py --configs "$C" --reps 2 --out runs/ab_pf.json > runs/ab_pf.log 2>&1
grep "median" runs/ab_pf.log

#!/bin/bash
# server smoke with spec decode: MIMO_SPEC=${1:-3}; holds the lease only for its duration
cd /data/Jarrel/mimo-pro-exl3-fast; source env.sh
trap 'kill $SP 2>/dev/null; sleep 5; kill -9 $SP 2>/dev/null; /data/Jarrel/coord/boxlease.sh release mimo' EXIT
MIMO_SPEC=${1:-3} MASTER_PORT=29546 setsid ./run_server.sh > runs/server_spec.log 2>&1 &
SP=$!
for i in $(seq 1 120); do curl -sf localhost:8003/health >/dev/null && break; sleep 5; done
grep "model ready" runs/server_spec.log
timeout 900 python test_server.py > runs/server_spec_test.log 2>&1; cat runs/server_spec_test.log
kill -- -$SP 2>/dev/null

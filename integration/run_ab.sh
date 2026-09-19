#!/bin/bash
# Paired A/B of vLLM's attention backend against ours, run from WSL2.
#
#   bash integration/run_ab.sh 7 5
#
# Argument 1 = pairs with the DEFAULT backend first  -> results/vllm_e2e.json
# Argument 2 = pairs with OURS first                 -> results/vllm_e2e_rev.json
#
# Both orderings are run because the second backend in a pair always sees a
# warmer GPU, and this machine also has sporadic multi-second stalls that can
# swallow an entire run (two baseline runs during data collection landed at
# 91 ms and 97 ms TPOT against a 15 ms median). A single A-then-B comparison
# here is worth nothing; `bench/analyze_ab.py` pairs the runs and reports the
# median paired difference with and without those outliers.
set -u
cd "$(dirname "$0")/.." || exit 1

FWD=${1:-7}
REV=${2:-5}

# Triton needs a host C compiler and CPython headers; integration/README.md has
# the no-root recipe that produces these two paths.
export CC=${CC:-$HOME/bin/zigcc}
export CPATH=${CPATH:-$HOME/pyhdr/usr/include/python3.10:$HOME/pyhdr/usr/include}
export PYTHONPATH=$PWD
PY=${PY:-$HOME/vllm126/bin/python}

COMMON="--model Qwen/Qwen2.5-0.5B-Instruct --requests 24 --output-tokens 96
        --max-model-len 1536 --gpu-memory-utilization 0.70"

rm -f results/vllm_e2e.json results/vllm_e2e_rev.json

for i in $(seq 1 "$FWD"); do
  timeout 1200 $PY integration/bench_vllm.py               $COMMON --out results/vllm_e2e.json >/dev/null 2>&1
  timeout 1200 $PY integration/bench_vllm.py --backend ours $COMMON --out results/vllm_e2e.json >/dev/null 2>&1
  echo "forward pair $i done"
done

for i in $(seq 1 "$REV"); do
  timeout 1200 $PY integration/bench_vllm.py --backend ours $COMMON --out results/vllm_e2e_rev.json >/dev/null 2>&1
  timeout 1200 $PY integration/bench_vllm.py               $COMMON --out results/vllm_e2e_rev.json >/dev/null 2>&1
  echo "reverse pair $i done"
done

python3 bench/analyze_ab.py 2>/dev/null || $PY bench/analyze_ab.py

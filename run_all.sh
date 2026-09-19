#!/usr/bin/env bash
# Everything except the main fp16 sweep, in the order the README presents it.
# GPU-exclusive steps run sequentially on purpose: overlapping them would make
# every timing in the report wrong.
set -u
cd "$(dirname "$0")"
step() { echo; echo "=================== $* ==================="; echo; }

step "1/9 correctness tests"
python -m pytest tests -q 2>&1 | tail -25

step "2/9 launch-overhead breakdown"
python -m bench.bench_overhead --out results/overhead.json 2>&1 | grep -viE "futurewarning|pynvml|^remark" | tail -30

step "3/9 KV-quantization sweep"
python -m bench.bench_decode --no-check --kv-dtypes fp16,fp8_e5m2,int8 \
  --batches 1,8,32 --seqlens 4096,16384 --impls triton,cuda \
  --out results/sweep_quant.json 2>&1 | grep -viE "futurewarning|pynvml|^remark" | tail -45

step "4/9 KV-quantization accuracy"
python -m bench.quant_accuracy --out results/quant_accuracy.json 2>&1 | grep -viE "futurewarning|pynvml|^remark" | tail -10

step "5/9 end-to-end decode loop"
# 1.0 GiB pool: the sdpa_memeff baseline must gather the paged KV *and* expand
# it to 32 heads every step, which peaks near 1.4 GiB on top of the pool.
python -m bench.bench_serving --steps 250 --max-batch 32 --kv-gib 1.0 --out results/serving.json 2>&1 \
  | grep -viE "futurewarning|pynvml|^remark" | tail -25

step "6/9 shape generality"
# Each point runs in its own process; see the header of bench/bench_shapes.py
# for why sharing a CUDA context between points produced wrong numbers.
python -m bench.bench_shapes --out results/shapes.json 2>&1 | grep -viE "futurewarning|pynvml|^remark" | tail -20

step "7/9 Nsight Systems"
powershell -ExecutionPolicy Bypass -File profiling/run_nsys.ps1 2>&1 | tail -15

step "8/9 Nsight Compute"
powershell -ExecutionPolicy Bypass -File profiling/run_ncu.ps1 2>&1 | tail -25

step "9/9 reports and figures"
python -m bench.report results/sweep_fp16.json --out results/TABLES.md 2>&1 | tail -3
python -m bench.report results/sweep_quant.json --out results/TABLES_quant.md 2>&1 | tail -3
python profiling/summarize_ncu.py results/ncu --out results/NCU.md 2>&1 | tail -3
python -m bench.plots results/sweep_fp16.json --outdir docs 2>&1 | tail -8
python -m bench.plots results/sweep_quant.json --outdir docs --only quantization 2>&1 | tail -2
# Cross-environment comparison, only if the WSL2 baselines have been run
# (see integration/README.md -- vLLM has no Windows build).
if [ -f results/vllm_kernels.json ]; then
  python -m bench.compare_vllm --out results/VLLM_COMPARISON.md 2>&1 | tail -3
else
  echo "skipping FA2/vLLM comparison: results/vllm_kernels.json not present"
fi
# The vLLM engine A/B is collected under WSL2 by integration/run_ab.sh.
if [ -f results/vllm_e2e.json ]; then
  python bench/analyze_ab.py 2>&1 | tail -14
else
  echo "skipping vLLM engine A/B: results/vllm_e2e.json not present"
fi
# Re-derive every headline README number from the JSON just written. A README
# is a cache of numbers that live elsewhere, and this one went stale twice.
python bench/check_readme.py 2>&1 | tail -6

echo "ALL_DONE"

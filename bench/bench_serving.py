"""End-to-end decode-loop benchmark: TPOT, throughput, throughput at a latency SLO.

    python -m bench.bench_serving --steps 400 --max-batch 32
    python -m bench.bench_serving --impls triton,cuda,sdpa_memeff --out results/serving.json

What this measures, precisely: the **attention portion** of a decode step, inside
a real continuous-batching loop -- ragged and growing sequence lengths, a live
block allocator, sequences retiring and being replaced, CUDA-graph replay with
bucketed batch sizes. It is not a full model forward; a Llama-3-8B's weights do
not fit in 4 GB, so quoting a whole-model TPOT here would be a number this
machine cannot produce. Every figure below is per-layer attention time, and the
README says how to scale it to a model.

That restriction does not weaken the comparison: attention is the only part of a
decode step whose cost grows with context length, and it is the only part any of
these implementations changes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import random
import statistics
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from bench.serving import (  # noqa: E402
    BlockAllocator, DecodeRunner, OutOfBlocks, Scheduler, poisson_lengths,
)
from bench.timing import measured_peak_gbs  # noqa: E402
from pagedattn import config as cfg  # noqa: E402
from pagedattn import cuda_decode, reference  # noqa: E402
from pagedattn.cache import gather_contiguous  # noqa: E402
from pagedattn.triton_decode import paged_decode_triton, pick_num_splits  # noqa: E402


def _kernel(impl: str):
    if impl == "triton":
        return lambda q, c, out, num_splits: paged_decode_triton(
            q, c, out=out, num_splits=num_splits)
    if impl == "cuda":
        return lambda q, c, out, num_splits: cuda_decode.paged_decode_cuda(
            q, c, out=out, num_splits=num_splits)
    if impl == "sdpa_memeff":
        # The dense baseline has to gather the paged KV into contiguous tensors
        # and expand the KV heads 4x every single step, because it understands
        # neither pages nor GQA. That gather is part of its cost and is timed.
        def run(q, c, out, num_splits):
            k, v = gather_contiguous(c)
            group = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(group, dim=1)
            v = v.repeat_interleave(group, dim=1)
            return reference.sdpa_decode(q, k, v, c.seq_lens, "mem_efficient")
        return run
    raise ValueError(impl)


@dataclasses.dataclass
class ServingResult:
    impl: str
    graphed: bool
    steps: int
    tpot_ms_p50: float
    tpot_ms_p90: float
    tpot_ms_p99: float
    tokens_per_s: float
    mean_batch: float
    mean_ctx: float
    max_ctx: int
    kv_utilization: float
    completed: int
    slo_attainment: dict[str, float]
    note: str = ""


def run_loop(
    impl: str,
    shape: cfg.ModelShape,
    num_blocks: int,
    block_size: int,
    max_batch: int,
    max_ctx: int,
    steps: int,
    seed: int,
    use_graphs: bool,
    slo_ms: list[float],
) -> ServingResult:
    rng = random.Random(seed)
    alloc = BlockAllocator(num_blocks)
    sched = Scheduler(
        alloc, block_size, max_batch,
        prompt_lens=poisson_lengths(rng, 128, max_ctx // 2),
        gen_lens=poisson_lengths(rng, 32, 512),
    )
    runner = DecodeRunner(
        shape, num_blocks, block_size, max_batch,
        max_blocks_per_seq=(max_ctx + block_size - 1) // block_size,
    )
    kernel = _kernel(impl)
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    buckets = runner.buckets()

    latencies: list[float] = []
    batch_sizes: list[int] = []
    ctx_lens: list[float] = []
    max_ctx_seen = 0
    util: list[float] = []
    graphed_any = False
    note = ""

    start_evt, end_evt = torch.cuda.Event(True), torch.cuda.Event(True)

    for _ in range(steps):
        try:
            sched.admit()
        except OutOfBlocks:
            pass
        if not sched.running:
            break

        seqs = sched.running
        n = len(seqs)
        bucket = next((b for b in buckets if b >= n), max_batch)
        runner.load_batch(seqs)

        max_seq = max(s.seq_len for s in seqs)
        ns = pick_num_splits(bucket, shape.num_kv_heads, max_seq, sm, 64)

        fn = None
        if use_graphs and impl in ("triton", "cuda"):
            g = runner.graph_for(bucket, kernel, ns)
            if callable(g):
                fn, graphed_any = g, True
            else:
                note = f"graph capture failed, running eager: {g}"
        if fn is None:
            fn = runner.make_fn(bucket, kernel, ns)

        torch.cuda.synchronize()
        start_evt.record()
        fn()
        end_evt.record()
        torch.cuda.synchronize()
        latencies.append(start_evt.elapsed_time(end_evt))

        batch_sizes.append(n)
        ctx_lens.append(sum(s.seq_len for s in seqs) / n)
        max_ctx_seen = max(max_ctx_seen, max_seq)
        util.append(alloc.utilization)
        sched.step_grow()

    if not latencies:
        raise RuntimeError("no steps ran")

    lat = sorted(latencies)
    total_tokens = sum(batch_sizes)
    total_s = sum(latencies) / 1e3

    def pct(p: float) -> float:
        return lat[min(len(lat) - 1, int(p * (len(lat) - 1)))]

    return ServingResult(
        impl=impl,
        graphed=graphed_any,
        steps=len(latencies),
        tpot_ms_p50=statistics.median(lat),
        tpot_ms_p90=pct(0.90),
        tpot_ms_p99=pct(0.99),
        tokens_per_s=total_tokens / total_s,
        mean_batch=statistics.mean(batch_sizes),
        mean_ctx=statistics.mean(ctx_lens),
        max_ctx=max_ctx_seen,
        kv_utilization=statistics.mean(util),
        completed=sched.completed,
        # Throughput you can actually promise: tokens/s counting only the steps
        # that met the deadline.
        slo_attainment={
            f"{s:g}ms": sum(1 for x in lat if x <= s) / len(lat) for s in slo_ms
        },
        note=note,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impls", default="triton,cuda,sdpa_memeff")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--kv-gib", type=float, default=1.6,
                    help="KV pool size; keep the pool + baselines inside VRAM")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--slo", default="0.5,1,2,4")
    ap.add_argument("--out", default="results/serving.json")
    args = ap.parse_args()

    shape = cfg.LLAMA3_8B
    per_token = cfg.kv_bytes_per_token(shape, "fp16")
    num_blocks = int(args.kv_gib * 2**30 / (per_token * args.block_size))
    slo_ms = [float(x) for x in args.slo.split(",") if x]

    info = cfg.gpu_info()
    print(json.dumps(info, indent=2))
    print(f"\nKV pool: {num_blocks:,} blocks x {args.block_size} tokens = "
          f"{num_blocks*args.block_size:,} tokens ({args.kv_gib:.2f} GiB)")
    print(f"max_batch={args.max_batch} max_ctx={args.max_ctx} steps={args.steps}\n")

    peak = measured_peak_gbs()
    results = []
    for impl in [x for x in args.impls.split(",") if x]:
        if impl == "cuda" and not cuda_decode.is_available():
            print(f"  {impl:12s} SKIP (extension unavailable: {cuda_decode.load_error()})")
            continue
        torch.cuda.empty_cache()
        try:
            r = run_loop(impl, shape, num_blocks, args.block_size, args.max_batch,
                         args.max_ctx, args.steps, args.seed,
                         use_graphs=not args.no_graphs, slo_ms=slo_ms)
        except Exception as exc:
            print(f"  {impl:12s} FAIL {type(exc).__name__}: {str(exc)[:120]}")
            continue
        results.append(r)
        slo = "  ".join(f"{k}:{v*100:.0f}%" for k, v in r.slo_attainment.items())
        print(f"  {impl:12s} graphed={str(r.graphed):5s} "
              f"TPOT p50={r.tpot_ms_p50:6.3f} p90={r.tpot_ms_p90:6.3f} "
              f"p99={r.tpot_ms_p99:6.3f} ms   {r.tokens_per_s:8,.0f} tok/s   "
              f"batch={r.mean_batch:4.1f} ctx={r.mean_ctx:6.0f}")
        print(f"               SLO attainment: {slo}   KV util={r.kv_utilization*100:.0f}%"
              + (f"   [{r.note}]" if r.note else ""))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "gpu": info,
        "peak_read_gbs_measured": peak,
        "config": vars(args),
        "kv_pool_blocks": num_blocks,
        "shape": dataclasses.asdict(shape),
        "results": [dataclasses.asdict(r) for r in results],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

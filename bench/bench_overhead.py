"""Isolate the three host-side costs that dominate a decode step at small batch.

    python -m bench.bench_overhead --out results/overhead.json

All four variants are measured in one process, back to back, on the same
allocations, because per-launch overhead on WDDM varies enough between processes
that comparing numbers from separate runs is meaningless. The variants are
cumulative:

  A  sync + alloc  : num_splits=None (forces a `.item()` device sync) and
                     out=None (allocates the output every call)  -- the naive
                     wrapper
  B  alloc only    : split count precomputed, output still allocated per call
  C  neither       : split count precomputed, output preallocated  -- eager,
                     the best a serving runtime can do without graphs
  D  CUDA graph    : C captured and replayed

D is what a real serving runtime pays. A is what a straightforward wrapper
costs. The gap between them is not kernel quality at all.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from bench.cudagraph import try_graph  # noqa: E402
from bench.timing import bench, measured_peak_gbs  # noqa: E402
from pagedattn import cache as cache_mod  # noqa: E402
from pagedattn import config as cfg  # noqa: E402
from pagedattn import cuda_decode  # noqa: E402
from pagedattn.triton_decode import paged_decode_triton, pick_num_splits  # noqa: E402


@triton.jit
def _noop_kernel(X):
    tl.store(X + tl.program_id(0), 0)


def launch_floor() -> dict[str, float]:
    """Cost of a launch that does no useful work, for scale."""
    x = torch.zeros(64, dtype=torch.int32, device="cuda")
    triton_ms = bench(lambda: _noop_kernel[(8,)](x))["ms"]
    g, _ = try_graph(lambda: _noop_kernel[(8,)](x))
    graph_ms = bench(g)["ms"] if g is not None else float("nan")
    sync_ms = bench(lambda: x.max().item())["ms"]
    return {
        "empty_triton_launch_us": triton_ms * 1e3,
        "empty_graph_replay_us": graph_ms * 1e3,
        "device_sync_item_us": sync_ms * 1e3,
    }


def measure_point(shape, batch, seqlen, impl, sm, peak) -> dict:
    c = cache_mod.allocate(shape, batch, seqlen, block_size=16, kv_dtype="fp16", seed=0)
    q = torch.randn(batch, shape.num_q_heads, shape.head_dim,
                    dtype=torch.float16, device="cuda")
    out = torch.empty_like(q)
    kv = c.bytes_read_per_decode_step()
    ns = pick_num_splits(batch, shape.num_kv_heads, seqlen, sm, 64 if impl == "triton" else 256)

    if impl == "triton":
        def call(o, splits):
            return paged_decode_triton(q, c, out=o, num_splits=splits)
    else:
        def call(o, splits):
            return cuda_decode.paged_decode_cuda(q, c, out=o, num_splits=splits)

    variants = {
        "A_sync_and_alloc": lambda: call(None, None),
        "B_alloc_only": lambda: call(None, ns),
        "C_neither": lambda: call(out, ns),
    }
    res = {k: bench(fn)["ms"] for k, fn in variants.items()}

    g, err = try_graph(lambda: call(out, ns))
    res["D_cuda_graph"] = bench(g)["ms"] if g is not None else float("nan")

    row = {
        "impl": impl, "batch": batch, "seqlen": seqlen, "num_splits": ns,
        "kv_bytes": kv,
        **{k: v * 1e3 for k, v in res.items()},  # microseconds
        "gbs_graph": kv / (res["D_cuda_graph"] * 1e-3) / 1e9,
        "pct_peak_graph": 100.0 * kv / (res["D_cuda_graph"] * 1e-3) / 1e9 / peak,
        "graph_note": "" if g is not None else err,
    }
    del c, q, out
    torch.cuda.empty_cache()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/overhead.json")
    ap.add_argument("--impls", default="triton,cuda")
    args = ap.parse_args()

    shape = cfg.LLAMA3_8B
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    info = cfg.gpu_info()
    print(json.dumps(info, indent=2))

    peak = measured_peak_gbs()
    print(f"measured peak: {peak:.1f} GB/s\n")

    floor = launch_floor()
    for k, v in floor.items():
        print(f"  {k:28s} {v:8.1f} us")
    print()

    hdr = (f"{'impl':7s} {'b':>3s} {'ctx':>6s} | {'A sync+alloc':>13s} "
           f"{'B alloc':>9s} {'C neither':>10s} {'D graph':>9s} | {'%peak(D)':>9s}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for impl in [x for x in args.impls.split(",") if x]:
        if impl == "cuda" and not cuda_decode.is_available():
            continue
        for batch, seqlen in [(1, 1024), (1, 4096), (1, 16384), (8, 4096), (32, 4096)]:
            r = measure_point(shape, batch, seqlen, impl, sm, peak)
            rows.append(r)
            print(f"{impl:7s} {batch:3d} {seqlen:6d} | {r['A_sync_and_alloc']:13.1f} "
                  f"{r['B_alloc_only']:9.1f} {r['C_neither']:10.1f} "
                  f"{r['D_cuda_graph']:9.1f} | {r['pct_peak_graph']:8.1f}%", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "gpu": info, "peak_read_gbs_measured": peak,
        "launch_floor_us": floor, "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

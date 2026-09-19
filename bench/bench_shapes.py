"""Both kernels across the attention shapes of real models.

    python -m bench.bench_shapes --out results/shapes.json

The main sweep fixes the shape and varies batch and context; this does the
opposite. It also measures, rather than asserts, how the two kernels differ in
generality: Triton takes group and head_dim as `constexpr` and specializes on
demand, while the CUDA kernel is instantiated ahead of time for a fixed list
(`cuda_decode.SUPPORTED_SHAPES`).

**Each point runs in its own subprocess.** Freeing the graph pool and calling
`empty_cache()` between points was not enough -- points measured after others
read as low as 35 % of peak where running them alone gave 97 %. A fresh CUDA
context per point costs a few seconds and removes the whole class of error.

`--one hq,hkv,d,batch` runs a single point; that is what the driver spawns.
"""

from __future__ import annotations

import argparse

import gc
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from bench.cudagraph import try_graph  # noqa: E402
from bench.timing import bench, measured_peak_gbs_checked  # noqa: E402
from pagedattn import cache as cache_mod  # noqa: E402
from pagedattn import config as cfg  # noqa: E402
from pagedattn import cuda_decode  # noqa: E402
from pagedattn.triton_decode import paged_decode_triton, pick_num_splits  # noqa: E402

# (num_q_heads, num_kv_heads, head_dim, label) for models that exist.
SHAPES = [
    (32, 8, 128, "Llama-3-8B"),
    (64, 8, 128, "Llama-3-70B"),
    (32, 8, 64, "Llama-3.2-1B"),
    (14, 2, 64, "Qwen2.5-0.5B"),
    (16, 2, 64, "Qwen2.5-3B-ish"),
]


def _free() -> None:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _measure(fn, kv_bytes: int, peak: float, repeats: int = 3) -> dict:
    """Best of `repeats` full measurements.

    `bench()` medians over many reps, but this machine has sporadic stalls that
    swallow a whole measurement window. Interference can only make a kernel look
    slower, so the minimum across independent measurements is the right
    estimator.
    """
    best = None
    for _ in range(repeats):
        graphed, note = try_graph(fn)
        res = bench(graphed if graphed is not None else fn)
        graphed = None      # release the graph's private pool
        _free()
        if best is None or res["ms"] < best[0]:
            best = (res["ms"], note == "")
    assert best is not None
    ms, graphed_ok = best
    gbs = kv_bytes / (ms * 1e-3) / 1e9
    return {"ms": ms, "gbs": gbs, "pct_peak": 100 * gbs / peak,
            "graphed": graphed_ok, "repeats": repeats}


def run_one(hq: int, hkv: int, d: int, batch: int, seqlen: int,
            peak: float | None = None) -> dict:
    """One point, in a process of its own. Prints a JSON line on stdout.

    `peak` comes from the parent. Letting each child probe it is wrong: a child
    starts while the previous one is still releasing the GPU, so the probe reads
    low and every "% of peak" is inflated -- one point reported 187 %.
    """
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    if peak is None:
        peak = measured_peak_gbs_checked()
    shape = cfg.ModelShape(hq, hkv, d, "one")
    group = hq // hkv
    c = cache_mod.allocate(shape, batch, seqlen, kv_dtype="fp16", seed=0)
    q = torch.randn(batch, hq, d, dtype=torch.float16, device="cuda")
    out = torch.empty_like(q)
    kv = c.bytes_read_per_decode_step()
    ns = pick_num_splits(batch, hkv, seqlen, sm, 64)

    row = {"num_q_heads": hq, "num_kv_heads": hkv, "head_dim": d, "group": group,
           "batch": batch, "seqlen": seqlen, "kv_bytes": kv, "num_splits": ns,
           "peak_gbs": peak}
    row["triton"] = _measure(
        lambda: paged_decode_triton(q, c, out=out, num_splits=ns), kv, peak)
    ok, why = cuda_decode.supports_shape(d, group)
    if ok and cuda_decode.is_available():
        row["cuda"] = _measure(
            lambda: cuda_decode.paged_decode_cuda(q, c, out=out, num_splits=ns),
            kv, peak)
    else:
        row["cuda_skipped"] = why or "extension unavailable"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="8,32")
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--one", default=None,
                    help="hq,hkv,d,batch -- run a single point and emit JSON")
    ap.add_argument("--peak", type=float, default=None,
                    help="bandwidth ceiling supplied by the parent process")
    ap.add_argument("--out", default="results/shapes.json")
    args = ap.parse_args()

    if args.one:
        hq, hkv, d, batch = (int(x) for x in args.one.split(","))
        print("__JSON__" + json.dumps(
            run_one(hq, hkv, d, batch, args.seqlen, args.peak)))
        return

    import subprocess
    import time

    # Probe the ceiling once, with the GPU idle.
    peak = measured_peak_gbs_checked()
    print(f"measured peak {peak:.1f} GB/s")

    hdr = (f"{'model':16s} {'q/kv/d':>11s} {'grp':>4s} {'b':>3s} | "
           f"{'triton':>18s} | {'cuda':>18s}")
    print(f"context {args.seqlen:,}; each point in a fresh process\n")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for hq, hkv, d, name in SHAPES:
        for batch in [int(x) for x in args.batches.split(",") if x]:
            time.sleep(2.0)   # let the previous child release the GPU
            res = subprocess.run(
                [sys.executable, "-m", "bench.bench_shapes",
                 "--one", f"{hq},{hkv},{d},{batch}", "--seqlen", str(args.seqlen),
                 "--peak", repr(peak)],
                capture_output=True, text=True,
                cwd=str(pathlib.Path(__file__).resolve().parent.parent))
            line = next((l for l in res.stdout.splitlines()
                         if l.startswith("__JSON__")), None)
            if line is None:
                print(f"{name:16s} {f'{hq}/{hkv}/{d}':>11s} {hq//hkv:4d} {batch:3d} | "
                      f"FAILED: {res.stderr.strip().splitlines()[-1][:50] if res.stderr.strip() else '?'}")
                continue
            row = json.loads(line[len("__JSON__"):])
            row["model"] = name
            rows.append(row)
            t = row["triton"]
            cu = row.get("cuda")
            c2 = (f"{cu['ms']*1e3:8.0f} us {cu['pct_peak']:5.1f}%" if cu
                  else f"{'not compiled':>18s}")
            print(f"{name:16s} {f'{hq}/{hkv}/{d}':>11s} {row['group']:4d} {batch:3d} | "
                  f"{t['ms']*1e3:8.0f} us {t['pct_peak']:5.1f}% | {c2:>18s}", flush=True)

    out_p = pathlib.Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(
        {"gpu": cfg.gpu_info(), "peak_read_gbs_measured": peak,
         "seqlen": args.seqlen,
         "cuda_supported_shapes": sorted(cuda_decode.SUPPORTED_SHAPES),
         "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out_p}")


if __name__ == "__main__":
    main()

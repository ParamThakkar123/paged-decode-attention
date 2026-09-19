"""Decode-attention sweep: latency, achieved DRAM bandwidth, tokens/sec.

Usage
-----
    python -m bench.bench_decode --out results/sweep.json
    python -m bench.bench_decode --quick                  # smoke-sized sweep
    python -m bench.bench_decode --impls triton,cuda      # subset
    python -m bench.bench_decode --kv-dtypes fp16,fp8_e5m2,int8

Every row records the VRAM the point needed and whether the KV working set fit
in L2, so no result has to be taken on faith.

Bandwidth, not FLOPs: decode does 2*S*D MACs per head against 2*S*D bytes of
KV, so arithmetic intensity is under 1 FLOP/byte at any context length and DRAM
is the only ceiling that matters.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import pathlib
import platform
import sys
import time
from typing import Callable

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from bench.cudagraph import try_graph  # noqa: E402
from bench.timing import (  # noqa: E402
    bench, l2_bytes, measured_peak_gbs_checked, theoretical_peak_gbs)
from pagedattn import cache as cache_mod  # noqa: E402
from pagedattn import config as cfg  # noqa: E402
from pagedattn import cuda_decode, reference  # noqa: E402
from pagedattn.triton_decode import paged_decode_triton, pick_num_splits  # noqa: E402

DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32, 64]
DEFAULT_SEQLENS = [1024, 2048, 4096, 8192, 16384, 32768]

# Paged implementations, run against the paged cache.
PAGED_IMPLS = [
    "triton",  # tuned: split-KV + per-page block table
    "triton_nosplit",  # ablation: no split-KV
    "triton_pertoken_bt",  # ablation: block table gathered per token
    "cuda",  # hand-written CUDA, split-KV, best variant (= v2_unroll)
    "cuda:v1_naive",  # ablation: 4 warps, 2 loads in flight per thread
    "cuda:v3_tuned",  # ablation: 8 warps, 8 loads in flight
    "cuda_nosplit",
]
# Dense baselines, run against gathered contiguous KV. `sdpa_cudnn` and
# `sdpa_math` take GQA shapes via enable_gqa; the other two reject them and
# need the KV heads expanded 8 -> 32 first, which is 4x the memory.
DENSE_IMPLS = ["sdpa_math", "sdpa_cudnn", "sdpa_memeff", "sdpa_flash"]
NEEDS_HEAD_EXPANSION = {"sdpa_memeff", "sdpa_flash"}


@dataclasses.dataclass
class Row:
    impl: str
    batch: int
    seqlen: int
    kv_dtype: str
    block_size: int
    ok: bool
    ms: float | None = None  # graphed if available, else eager -- the headline number
    ms_eager: float | None = None
    ms_graph: float | None = None
    ms_p20: float | None = None
    ms_p80: float | None = None
    gbs: float | None = None
    pct_peak: float | None = None
    tok_per_s: float | None = None
    launch_overhead_ms: float | None = None
    peak_gbs: float | None = None  # ceiling measured closest in time to this row
    kv_bytes: int | None = None
    peak_vram_bytes: int | None = None
    num_splits: int | None = None
    l2_resident_frac: float | None = None
    max_rel_err: float | None = None
    note: str = ""


def _free_all() -> None:
    """Best-effort reclaim between sweep points.

    Wrapped because `empty_cache()` itself raises on a context that has OOMed,
    and one bad point must not take the whole sweep down with it.
    """
    try:
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception as exc:  # pragma: no cover - only on a poisoned context
        print(f"    [warn] cleanup failed: {type(exc).__name__}: {exc}", flush=True)


def _dense_bytes(shape: cfg.ModelShape, batch: int, seqlen: int, expanded: bool) -> int:
    """VRAM the dense baselines need for their contiguous KV.

    The fused SDPA backends reject GQA, so their KV heads are expanded 8 -> 32
    -- 4x the memory. A property of the baseline, not the harness, and why they
    OOM at points the paged kernels handle.
    """
    heads = shape.num_q_heads if expanded else shape.num_kv_heads
    kv = 2 * batch * heads * seqlen * shape.head_dim * 2
    if expanded:
        kv += 2 * batch * shape.num_kv_heads * seqlen * shape.head_dim * 2  # the source
    return kv


def _run_paged(
    impl: str, q: torch.Tensor, out: torch.Tensor, c: cache_mod.PagedKVCache, sm_count: int
):
    """Return (callable, num_splits) for a paged implementation.

    `out` preallocated and `num_splits` precomputed keep the timed closure free
    of host work and allocation -- what a runtime does, and what graph capture
    requires.
    """
    max_seq = int(c.seq_lens.max().item())
    b = c.batch
    hkv = c.shape.num_kv_heads
    ns_tri = pick_num_splits(b, hkv, max_seq, sm_count, 64)
    ns_cuda = pick_num_splits(b, hkv, max_seq, sm_count, 256)

    if impl == "triton":
        return (lambda: paged_decode_triton(q, c, out=out, num_splits=ns_tri,
                                            per_page_bt=True)), ns_tri
    if impl == "triton_nosplit":
        return (lambda: paged_decode_triton(q, c, out=out, num_splits=1,
                                            per_page_bt=True)), 1
    if impl == "triton_pertoken_bt":
        return (lambda: paged_decode_triton(q, c, out=out, num_splits=ns_tri,
                                            per_page_bt=False)), ns_tri
    if impl.startswith("cuda"):
        variant = impl.split(":", 1)[1] if ":" in impl else cuda_decode.DEFAULT_VARIANT
        ns = 1 if impl.startswith("cuda_nosplit") else ns_cuda
        return (lambda: cuda_decode.paged_decode_cuda(q, c, out=out, num_splits=ns,
                                                      variant=variant)), ns
    raise ValueError(impl)


def _measure(fn, kv_bytes: int, batch: int, peak_gbs: float,
             allow_graph: bool = True) -> dict:
    """Time `fn` eagerly and under graph replay; report the better one.

    Graphed wins when capture succeeds, because the eager number on WDDM is
    dominated by launch overhead a serving runtime does not pay. Both are kept;
    the difference is the launch-overhead column.
    """
    eager = bench(fn)["ms"]

    graphed, note = (None, "graphing disabled") if not allow_graph else try_graph(fn)
    captured = graphed is not None

    if captured:
        res = bench(graphed)
        ms_graph = res["ms"]
    else:
        res = bench(fn)
        ms_graph = None

    # A captured graph owns a private pool the allocator reclaims only when the
    # graph dies, and even then keeps the blocks reserved. Over several hundred
    # captures those pools accumulate until the largest points run under memory
    # pressure -- this turned a 102 %-of-peak point into a 52 % one.
    graphed = None
    gc.collect()
    torch.cuda.empty_cache()

    ms, p20, p80 = res["ms"], res["ms_p20"], res["ms_p80"]
    gbs = kv_bytes / (ms * 1e-3) / 1e9
    return {
        "ms": ms, "ms_eager": eager, "ms_graph": ms_graph,
        "ms_p20": p20, "ms_p80": p80,
        "gbs": gbs, "pct_peak": 100.0 * gbs / peak_gbs,
        "tok_per_s": batch / (ms * 1e-3),
        "launch_overhead_ms": (eager - ms_graph) if ms_graph is not None else None,
        "note": "" if captured else f"not graphed: {note}",
    }


def sweep(
    batches: list[int],
    seqlens: list[int],
    kv_dtypes: list[str],
    impls: list[str],
    block_size: int,
    shape: cfg.ModelShape,
    budget: cfg.VRamBudget,
    peak_gbs: float,
    check: bool,
    shuffle: bool,
    checkpoint: Callable[[list[Row]], None] | None = None,
) -> list[Row]:
    rows: list[Row] = []
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    l2 = l2_bytes()

    paged_impls = [i for i in impls if i in PAGED_IMPLS]
    dense_impls = [i for i in impls if i in DENSE_IMPLS]

    for kv_dtype in kv_dtypes:
        for seqlen in seqlens:
            # Re-measure the ceiling per context block. This 70 W laptop part
            # hits `SW Power Cap: Active` during a long sweep and settles at
            # lower clocks, so a cold-card ceiling understates later points --
            # it made saturated points look like 69-88%.
            _free_all()
            peak_gbs = measured_peak_gbs_checked()
            print(f"  [ctx {seqlen}] re-measured peak: {peak_gbs:.1f} GB/s", flush=True)

            for batch in batches:
                fits, need = budget.fits(shape, batch, seqlen, kv_dtype, block_size)  # type: ignore[arg-type]
                tag = f"{kv_dtype} b={batch:<3d} s={seqlen:<6d}"
                if not fits:
                    for impl in impls:
                        rows.append(Row(impl, batch, seqlen, kv_dtype, block_size, False,
                                        kv_bytes=need,
                                        note=f"skipped: KV {need/2**30:.2f} GiB over "
                                             f"{budget.kv_budget_bytes/2**30:.2f} GiB budget"))
                    print(f"  {tag}  SKIP (needs {need/2**30:.2f} GiB)", flush=True)
                    continue

                # ---------------- paged phase --------------------------------
                _free_all()
                try:
                    c = cache_mod.allocate(shape, batch, seqlen, block_size=block_size,
                                           kv_dtype=kv_dtype, shuffle=shuffle, seed=0)  # type: ignore[arg-type]
                    q = torch.randn(batch, shape.num_q_heads, shape.head_dim,
                                    dtype=torch.float16, device="cuda")
                    out_buf = torch.empty_like(q)
                except torch.cuda.OutOfMemoryError as exc:
                    for impl in impls:
                        rows.append(Row(impl, batch, seqlen, kv_dtype, block_size, False,
                                        kv_bytes=need, note=f"OOM on allocate: {exc}"[:200]))
                    print(f"  {tag}  OOM on allocate", flush=True)
                    _free_all()
                    continue

                kv_bytes = c.bytes_read_per_decode_step()
                l2_frac = min(1.0, l2 / kv_bytes)
                ref = reference.reference_decode(q, c) if check else None

                for impl in paged_impls:
                    try:
                        fn, ns = _run_paged(impl, q, out_buf, c, sm_count)
                        err = None
                        if ref is not None:
                            got = fn()
                            denom = ref.float().abs().max().item()
                            err = (got.float() - ref.float()).abs().max().item() / max(denom, 1e-6)
                        m = _measure(fn, kv_bytes, batch, peak_gbs)
                        rows.append(Row(
                            impl, batch, seqlen, kv_dtype, block_size, True,
                            ms=m["ms"], ms_eager=m["ms_eager"], ms_graph=m["ms_graph"],
                            ms_p20=m["ms_p20"], ms_p80=m["ms_p80"],
                            gbs=m["gbs"], pct_peak=m["pct_peak"], tok_per_s=m["tok_per_s"],
                            launch_overhead_ms=m["launch_overhead_ms"],
                            kv_bytes=kv_bytes,
                            peak_vram_bytes=torch.cuda.max_memory_allocated(),
                            num_splits=ns, l2_resident_frac=l2_frac, max_rel_err=err,
                            peak_gbs=peak_gbs, note=m["note"],
                        ))
                        print(f"  {tag}  {impl:20s} {m['ms']*1e3:8.1f} us  "
                              f"{m['gbs']:6.1f} GB/s  {m['pct_peak']:5.1f}% peak"
                              f"  (eager {m['ms_eager']*1e3:7.1f})"
                              + (f"  rel={err:.1e}" if err is not None else ""), flush=True)
                    except Exception as exc:
                        rows.append(Row(impl, batch, seqlen, kv_dtype, block_size, False,
                                        kv_bytes=kv_bytes,
                                        note=f"{type(exc).__name__}: {exc}"[:200]))
                        print(f"  {tag}  {impl:20s} FAIL {type(exc).__name__}: "
                              f"{str(exc)[:90]}", flush=True)

                del q, out_buf
                if ref is not None:
                    del ref
                seq_lens = c.seq_lens.clone()
                del c
                _free_all()

                # ---------------- dense phase --------------------------------
                # Separate, so gathered KV never coexists with the paged cache;
                # on 4 GB that alone decides whether a point runs.
                if checkpoint is not None:
                    checkpoint(rows)

                if dense_impls and kv_dtype in ("fp16", "bf16"):
                    _run_dense(dense_impls, shape, batch, seqlen, kv_dtype, block_size,
                               seq_lens, kv_bytes, peak_gbs, l2_frac, rows, tag, budget)
                    _free_all()

                if checkpoint is not None:
                    checkpoint(rows)
    return rows


def _run_dense(dense_impls, shape, batch, seqlen, kv_dtype, block_size, seq_lens,
               kv_bytes, peak_gbs, l2_frac, rows, tag, budget) -> None:
    hq, hkv, d = shape.num_q_heads, shape.num_kv_heads, shape.head_dim
    group = hq // hkv

    # Check before allocating: an OOM mid-capture poisons the context badly
    # enough that empty_cache() then raises too, which took out a whole sweep.
    need_plain = _dense_bytes(shape, batch, seqlen, expanded=False)
    need_expanded = _dense_bytes(shape, batch, seqlen, expanded=True)
    ceiling = budget.kv_budget_bytes
    if need_plain > ceiling:
        for impl in dense_impls:
            rows.append(Row(impl, batch, seqlen, kv_dtype, block_size, False,
                            kv_bytes=kv_bytes,
                            note=f"skipped: dense KV {need_plain/2**30:.2f} GiB over "
                                 f"{ceiling/2**30:.2f} GiB budget"))
        print(f"  {tag}  dense SKIP (needs {need_plain/2**30:.2f} GiB)", flush=True)
        return
    fused_ok = need_expanded <= ceiling
    if not fused_ok:
        for impl in [i for i in dense_impls if i in NEEDS_HEAD_EXPANSION]:
            rows.append(Row(impl, batch, seqlen, kv_dtype, block_size, False,
                            kv_bytes=kv_bytes,
                            note=f"skipped: 4x KV-head expansion needs "
                                 f"{need_expanded/2**30:.2f} GiB, over "
                                 f"{ceiling/2**30:.2f} GiB budget"))
        dense_impls = [i for i in dense_impls if i not in NEEDS_HEAD_EXPANSION]
        if not dense_impls:
            print(f"  {tag}  fused-SDPA SKIP (head expansion needs "
                  f"{need_expanded/2**30:.2f} GiB)", flush=True)
            return

    try:
        q = torch.randn(batch, hq, d, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, hkv, seqlen, d, dtype=torch.float16, device="cuda")
        v = torch.randn_like(k)
    except torch.cuda.OutOfMemoryError as exc:
        for impl in dense_impls:
            rows.append(Row(impl, batch, seqlen, kv_dtype, block_size, False,
                            note=f"OOM allocating dense KV: {exc}"[:160]))
        return

    kx = vx = None
    for impl in dense_impls:
        try:
            if impl == "sdpa_math":
                fn = lambda: reference.sdpa_decode(q, k, v, seq_lens, "math")  # noqa: E731
                extra = 0
            elif impl == "sdpa_cudnn":
                # Native GQA, no mask, no expansion.
                fn = lambda: reference.sdpa_gqa_uniform(q, k, v, "cudnn")  # noqa: E731
                extra = 0
            else:
                # These backends reject GQA, so expand the KV heads -- 4x the
                # memory.
                if kx is None:
                    kx = k.repeat_interleave(group, dim=1).contiguous()
                    vx = v.repeat_interleave(group, dim=1).contiguous()
                backend = "mem_efficient" if impl == "sdpa_memeff" else "flash"
                fn = lambda b=backend: reference.sdpa_decode(q, kx, vx, seq_lens, b)  # noqa: E731
                extra = 2 * kx.numel() * 2
            fn()
            # These allocate inside the timed region, so a capture would pin a
            # graph pool per point and never release it.
            m = _measure(fn, kv_bytes, batch, peak_gbs, allow_graph=False)
            extra_note = (f"dense KV; +{extra/2**20:.0f} MiB head expansion"
                          if extra else "dense KV")
            rows.append(Row(
                impl, batch, seqlen, kv_dtype, block_size, True,
                ms=m["ms"], ms_eager=m["ms_eager"], ms_graph=m["ms_graph"],
                ms_p20=m["ms_p20"], ms_p80=m["ms_p80"],
                gbs=m["gbs"], pct_peak=m["pct_peak"], tok_per_s=m["tok_per_s"],
                launch_overhead_ms=m["launch_overhead_ms"],
                kv_bytes=kv_bytes, peak_vram_bytes=torch.cuda.max_memory_allocated(),
                num_splits=None, l2_resident_frac=l2_frac, peak_gbs=peak_gbs,
                note=(extra_note + ("; " + m["note"] if m["note"] else "")),
            ))
            print(f"  {tag}  {impl:20s} {m['ms']*1e3:8.1f} us  {m['gbs']:6.1f} GB/s  "
                  f"{m['pct_peak']:5.1f}% peak  (eager {m['ms_eager']*1e3:7.1f})", flush=True)
        except Exception as exc:
            rows.append(Row(impl, batch, seqlen, kv_dtype, block_size, False,
                            kv_bytes=kv_bytes, note=f"{type(exc).__name__}: {exc}"[:200]))
            print(f"  {tag}  {impl:20s} FAIL {type(exc).__name__}: {str(exc)[:90]}", flush=True)
    del q, k, v, kx, vx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/sweep.json")
    ap.add_argument("--batches", default=",".join(map(str, DEFAULT_BATCHES)))
    ap.add_argument("--seqlens", default=",".join(map(str, DEFAULT_SEQLENS)))
    ap.add_argument("--kv-dtypes", default="fp16")
    ap.add_argument("--impls", default=",".join(PAGED_IMPLS + DENSE_IMPLS))
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--kv-fraction", type=float, default=0.55,
                    help="fraction of total VRAM the KV cache may occupy")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the per-point correctness check (faster)")
    ap.add_argument("--sequential-blocks", action="store_true",
                    help="allocate block tables sequentially instead of fragmented")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.batches, args.seqlens = "1,8,32", "1024,4096,16384"

    batches = [int(x) for x in args.batches.split(",") if x]
    seqlens = [int(x) for x in args.seqlens.split(",") if x]
    kv_dtypes = [x for x in args.kv_dtypes.split(",") if x]
    impls = [x for x in args.impls.split(",") if x]

    info = cfg.gpu_info()
    print(json.dumps(info, indent=2))

    if any(i.startswith("cuda") for i in impls):
        if not cuda_decode.is_available():
            print(f"!! CUDA extension unavailable, dropping cuda impls: "
                  f"{cuda_decode.load_error()}")
            impls = [i for i in impls if not i.startswith("cuda")]

    print("measuring peak read bandwidth ...", flush=True)
    peak = measured_peak_gbs_checked()
    theo = theoretical_peak_gbs()
    print(f"  measured streaming-read peak: {peak:.1f} GB/s"
          + (f"   (theoretical {theo:.1f} GB/s)" if theo else "   (theoretical: n/a)"))

    budget = cfg.VRamBudget.from_device(kv_fraction=args.kv_fraction)
    print(f"  KV budget: {budget.kv_budget_bytes/2**30:.2f} GiB of "
          f"{budget.total_bytes/2**30:.2f} GiB\n")

    t0 = time.time()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def payload_for(rows: list[Row]) -> dict:
        return {
            "gpu": info,
            "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
            "peak_read_gbs_measured": peak,
            "peak_gbs_theoretical": theo,
            "l2_bytes": l2_bytes(),
            "shape": dataclasses.asdict(cfg.LLAMA3_8B),
            "block_size": args.block_size,
            "block_table": "sequential" if args.sequential_blocks else "fragmented",
            "elapsed_s": round(time.time() - t0, 1),
            "rows": [dataclasses.asdict(r) for r in rows],
        }

    def checkpoint(rows: list[Row]) -> None:
        """Write after every point, so one OOM at the end cannot cost the whole
        sweep."""
        out.write_text(json.dumps(payload_for(rows), indent=2), encoding="utf-8")

    rows = sweep(batches, seqlens, kv_dtypes, impls, args.block_size,
                 cfg.LLAMA3_8B, budget, peak, check=not args.no_check,
                 shuffle=not args.sequential_blocks, checkpoint=checkpoint)

    payload = payload_for(rows)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {len(rows)} rows to {out}  ({payload['elapsed_s']} s)")


if __name__ == "__main__":
    main()

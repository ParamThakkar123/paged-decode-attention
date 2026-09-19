"""FlashInfer paged-decode baseline.

    # inside WSL2, in the cu126 environment (see integration/README.md):
    ~/vllm126/bin/python -m bench.flashinfer_baseline --out results/flashinfer.json

NOTE: unused on this machine. FlashInfer JIT-compiles with `nvcc`, which the
WSL image does not have (`nvidia-cuda-nvcc-cu12` ships `ptxas` only) and cannot
install without root. `bench/vllm_kernels_baseline.py` covers the same ground
with two precompiled paged kernels. Kept because it is correct and runs
anywhere `nvcc` exists.

FlashInfer takes block tables in CSR form rather than a dense 2D table:

    indptr[i]        first index into `indices` belonging to sequence i
    indices[...]     the flat concatenation of every sequence's page ids
    last_page_len[i] how many of the last page's slots are real tokens

`to_flashinfer_layout` converts ours into that -- a pure index rearrangement,
since the KV tensors are already in the NHD layout FlashInfer wants.
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from bench.timing import bench, measured_peak_gbs  # noqa: E402
from pagedattn import cache as cache_mod  # noqa: E402
from pagedattn import config as cfg  # noqa: E402
from pagedattn.cache import PagedKVCache  # noqa: E402
from pagedattn.triton_decode import paged_decode_triton, pick_num_splits  # noqa: E402


def to_flashinfer_layout(cache: PagedKVCache) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Our dense block table -> FlashInfer's (indptr, indices, last_page_len)."""
    bs = cache.block_size
    seq_lens = cache.seq_lens.to(torch.int32)
    pages_per_seq = torch.ceil(seq_lens.float() / bs).to(torch.int32)

    indptr = torch.zeros(cache.batch + 1, dtype=torch.int32, device=seq_lens.device)
    indptr[1:] = torch.cumsum(pages_per_seq, dim=0)

    indices = torch.cat([
        cache.block_table[i, : int(pages_per_seq[i])] for i in range(cache.batch)
    ]).to(torch.int32)

    # A full last page is `bs`, not 0.
    last = seq_lens - (pages_per_seq - 1) * bs
    return indptr, indices, last.to(torch.int32)


def _run_point(shape, batch, seqlen, block_size, workspace, sm, peak) -> dict:
    """One (batch, context) point. A function so every tensor it allocates goes
    out of scope on return -- on a 4 GiB card the previous point's KV cache is
    exactly what makes the next one fail to allocate."""
    c = cache_mod.allocate(shape, batch, seqlen, block_size=block_size,
                           kv_dtype="fp16", shuffle=True, seed=0)
    q = torch.randn(batch, shape.num_q_heads, shape.head_dim,
                    dtype=torch.float16, device="cuda")
    out = torch.empty_like(q)
    kv_bytes = c.bytes_read_per_decode_step()
    ns = pick_num_splits(batch, shape.num_kv_heads, seqlen, sm, 64)

    wrapper = build_wrapper(c, shape, workspace)
    fi_out = wrapper.run(q, (c.k_cache, c.v_cache))
    ours = paged_decode_triton(q, c, out=out, num_splits=ns)
    rel = ((fi_out.float() - ours.float()).abs().max()
           / ours.float().abs().max().clamp_min(1e-6)).item()

    t_ours = bench(lambda: paged_decode_triton(q, c, out=out, num_splits=ns))["ms"]
    t_fi = bench(lambda: wrapper.run(q, (c.k_cache, c.v_cache)))["ms"]

    g_ours = kv_bytes / (t_ours * 1e-3) / 1e9
    g_fi = kv_bytes / (t_fi * 1e-3) / 1e9
    return {
        "batch": batch, "seqlen": seqlen, "kv_bytes": kv_bytes,
        "ours_ms": t_ours, "flashinfer_ms": t_fi,
        "ours_pct_peak": 100 * g_ours / peak,
        "flashinfer_pct_peak": 100 * g_fi / peak,
        "ratio_fi_over_ours": t_fi / t_ours,
        "max_rel_diff": rel,
    }


def build_wrapper(cache: PagedKVCache, shape: cfg.ModelShape, workspace: torch.Tensor):
    import flashinfer

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
    indptr, indices, last_page_len = to_flashinfer_layout(cache)
    wrapper.plan(
        indptr, indices, last_page_len,
        shape.num_q_heads, shape.num_kv_heads, shape.head_dim,
        cache.block_size,
        pos_encoding_mode="NONE",
        data_type=cache.k_cache.dtype,
        q_data_type=torch.float16,
    )
    return wrapper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,2,4,8,16,32,64")
    ap.add_argument("--seqlens", default="1024,4096,16384")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--kv-fraction", type=float, default=0.45,
                    help="lower than the sweep's: FlashInfer keeps its own workspace")
    ap.add_argument("--out", default="results/flashinfer.json")
    args = ap.parse_args()

    try:
        import flashinfer  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"flashinfer not importable: {exc}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable -- see integration/README.md 'driver trap'")
    # Import success is not enough; a cu130 torch on a 12.6 driver imports fine
    # and then fails on the first launch.
    torch.randn(8, device="cuda").sum().item()

    shape = cfg.LLAMA3_8B
    budget = cfg.VRamBudget.from_device(kv_fraction=args.kv_fraction)
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    peak = measured_peak_gbs()
    print(f"measured peak {peak:.1f} GB/s, KV budget "
          f"{budget.kv_budget_bytes / 2**30:.2f} GiB\n")

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    rows = []
    hdr = f"{'b':>3} {'ctx':>7} | {'ours us':>9} {'flashinfer us':>14} | {'ours %pk':>9} {'fi %pk':>8} {'ratio':>7}"
    print(hdr)
    print("-" * len(hdr))

    for seqlen in [int(x) for x in args.seqlens.split(",") if x]:
        for batch in [int(x) for x in args.batches.split(",") if x]:
            fits, need = budget.fits(shape, batch, seqlen, "fp16", args.block_size)
            if not fits:
                print(f"{batch:3d} {seqlen:7d} | SKIP ({need / 2**30:.2f} GiB)")
                continue
            try:
                r = _run_point(shape, batch, seqlen, args.block_size,
                               workspace, sm, peak)
                rows.append(r)
                print(f"{batch:3d} {seqlen:7d} | {r['ours_ms']*1e3:9.0f} "
                      f"{r['flashinfer_ms']*1e3:14.0f} | {r['ours_pct_peak']:8.1f}% "
                      f"{r['flashinfer_pct_peak']:7.1f}% "
                      f"{r['ratio_fi_over_ours']:6.2f}x   rel={r['max_rel_diff']:.1e}",
                      flush=True)
            except Exception as exc:
                print(f"{batch:3d} {seqlen:7d} | FAIL {type(exc).__name__}: {str(exc)[:90]}")
            finally:
                gc.collect()
                torch.cuda.empty_cache()

    out_p = pathlib.Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(
        {"gpu": cfg.gpu_info(), "peak_read_gbs_measured": peak, "rows": rows}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {out_p}")


if __name__ == "__main__":
    main()

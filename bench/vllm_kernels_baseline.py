"""Paged-decode baselines that ship precompiled inside vLLM.

    # inside WSL2, in the cu126 environment (see integration/README.md):
    cd /mnt/e/Projects/inference_benchmark
    ~/vllm126/bin/python -m bench.vllm_kernels_baseline --out results/vllm_kernels.json

Two baselines, both of which are the real thing rather than a dense stand-in:

  **FA2 paged decode** — `vllm_flash_attn.flash_attn_with_kvcache`, which is
  FlashAttention-2's paged-KV decode path. It consumes
  `[num_blocks, page_size, num_kv_heads, head_dim]` and a dense
  `[batch, max_blocks]` block table, i.e. *exactly* our layout, so there is no
  conversion and no conversion cost to argue about. This is the FA2 comparison
  that native Windows cannot run at all.

  **vLLM PagedAttention** — `vllm._custom_ops.paged_attention_v1/v2`, the V0
  kernel. It wants a different cache layout (K split into an `x`-vectorized
  minor axis, V transposed), so `_to_vllm_v0_layout` repacks once, outside the
  timed region. Repacking per call would be benchmarking a memcpy.

Both are chosen over FlashInfer for one practical reason: they are compiled into
the vLLM wheel, whereas FlashInfer JIT-compiles with `nvcc`, and this WSL install
has no CUDA toolkit. The brief allows either.
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from bench.timing import (  # noqa: E402
    bench, measured_peak_gbs, measured_peak_gbs_torch, triton_usable)
from pagedattn import cache as cache_mod  # noqa: E402
from pagedattn import config as cfg  # noqa: E402
from pagedattn.cache import PagedKVCache  # noqa: E402
try:
    from pagedattn.triton_decode import paged_decode_triton, pick_num_splits  # noqa: E402
    _IMPORTED = True
except Exception:
    _IMPORTED = False

# Resolved in main(), after CUDA is known good: importing Triton is not the same
# as Triton being able to compile. See bench.timing.triton_usable.
HAVE_TRITON = False

_PARTITION = 512  # vLLM's _PARTITION_SIZE for paged_attention_v2


def have(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def fa2_runner(q: torch.Tensor, c: PagedKVCache, shape: cfg.ModelShape):
    """FlashAttention-2 paged decode. Our layout needs no conversion."""
    from vllm.vllm_flash_attn import flash_attn_with_kvcache

    q4 = q.unsqueeze(1)  # [B, 1, H_q, D]: FA2 wants an explicit query-length axis

    def run():
        return flash_attn_with_kvcache(
            q=q4,
            k_cache=c.k_cache,
            v_cache=c.v_cache,
            cache_seqlens=c.seq_lens,
            block_table=c.block_table,
            softmax_scale=shape.softmax_scale,
            causal=False,
        )

    return run


def _to_vllm_v0_layout(c: PagedKVCache, shape: cfg.ModelShape):
    """Our NHD cache -> vLLM V0's split-K / transposed-V layout.

    K: [num_blocks, num_kv_heads, head_dim/x, page_size, x]   (x = 16 / elem_size)
    V: [num_blocks, num_kv_heads, head_dim, page_size]
    Done once, outside the timed region.
    """
    nb, ps, h, d = c.k_cache.shape
    x = 16 // c.k_cache.element_size()
    k = c.k_cache.permute(0, 2, 3, 1).contiguous()            # [nb, h, d, ps]
    k = k.view(nb, h, d // x, x, ps).permute(0, 1, 2, 4, 3).contiguous()
    v = c.v_cache.permute(0, 2, 3, 1).contiguous()            # [nb, h, d, ps]
    return k, v


def vllm_paged_runner(q: torch.Tensor, c: PagedKVCache, shape: cfg.ModelShape):
    """vLLM's own PagedAttention kernel (V0 custom op)."""
    import vllm._custom_ops as ops

    k, v = _to_vllm_v0_layout(c, shape)
    out = torch.empty_like(q)
    max_seq = int(c.seq_lens.max().item())
    scale = shape.softmax_scale
    k_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)

    num_partitions = (max_seq + _PARTITION - 1) // _PARTITION
    use_v2 = num_partitions > 1
    b, hq, d = q.shape
    if use_v2:
        exp_sums = torch.empty((b, hq, num_partitions), dtype=torch.float32, device=q.device)
        max_logits = torch.empty_like(exp_sums)
        tmp_out = torch.empty((b, hq, num_partitions, d), dtype=q.dtype, device=q.device)

        def run():
            ops.paged_attention_v2(
                out, exp_sums, max_logits, tmp_out, q, k, v,
                shape.num_kv_heads, scale, c.block_table, c.seq_lens,
                c.block_size, max_seq, None, "auto", k_scale, v_scale,
            )
            return out
    else:
        def run():
            ops.paged_attention_v1(
                out, q, k, v, shape.num_kv_heads, scale, c.block_table, c.seq_lens,
                c.block_size, max_seq, None, "auto", k_scale, v_scale,
            )
            return out

    return run, (k, v)


def _run_point(shape, batch, seqlen, block_size, sm, peak, impls) -> dict:
    c = cache_mod.allocate(shape, batch, seqlen, block_size=block_size,
                           kv_dtype="fp16", shuffle=True, seed=0)
    q = torch.randn(batch, shape.num_q_heads, shape.head_dim,
                    dtype=torch.float16, device="cuda")
    out = torch.empty_like(q)
    kv_bytes = c.bytes_read_per_decode_step()
    ns = pick_num_splits(batch, shape.num_kv_heads, seqlen, sm, 64) if HAVE_TRITON else 1

    row = {"batch": batch, "seqlen": seqlen, "kv_bytes": kv_bytes}
    ref = None
    t_ours = None
    if HAVE_TRITON:
        ours = paged_decode_triton(q, c, out=out, num_splits=ns)
        ref = ours.float().clone()
        t_ours = bench(lambda: paged_decode_triton(q, c, out=out, num_splits=ns))["ms"]
        row["ours_ms"] = t_ours
        row["ours_pct_peak"] = 100 * kv_bytes / (t_ours * 1e-3) / 1e9 / peak

    keep = []  # hold repacked caches alive for the duration of the timing
    for name in impls:
        try:
            if name == "fa2":
                run = fa2_runner(q, c, shape)
            else:
                run, packed = vllm_paged_runner(q, c, shape)
                keep.append(packed)
            got = run()
            t = bench(run)["ms"]
            row[f"{name}_ms"] = t
            row[f"{name}_gbs"] = kv_bytes / (t * 1e-3) / 1e9
            row[f"{name}_pct_peak"] = 100 * kv_bytes / (t * 1e-3) / 1e9 / peak
            if ref is not None and t_ours is not None:
                g2 = got.reshape(ref.shape).float()
                row[f"{name}_max_rel_diff"] = (
                    (g2 - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
                row[f"{name}_ratio_over_ours"] = t / t_ours
        except Exception as exc:
            row[f"{name}_error"] = f"{type(exc).__name__}: {exc}"[:160]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,2,4,8,16,32,64")
    ap.add_argument("--seqlens", default="1024,4096,16384")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--kv-fraction", type=float, default=0.40,
                    help="low: the vLLM V0 path keeps a second, repacked copy of the cache")
    ap.add_argument("--impls", default="fa2,vllm_paged")
    ap.add_argument("--out", default="results/vllm_kernels.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable -- see integration/README.md 'driver trap'")
    torch.randn(8, device="cuda").sum().item()  # a real launch, not just an import

    global HAVE_TRITON
    HAVE_TRITON = _IMPORTED and triton_usable()

    impls = [x for x in args.impls.split(",") if x]
    avail = {"fa2": have("vllm.vllm_flash_attn"), "vllm_paged": have("vllm._custom_ops")}
    for name in list(impls):
        if not avail.get(name):
            print(f"!! {name} unavailable, dropping")
            impls.remove(name)
    if not impls:
        raise SystemExit("no vLLM kernel baselines importable")

    shape = cfg.LLAMA3_8B
    budget = cfg.VRamBudget.from_device(kv_fraction=args.kv_fraction)
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    peak = measured_peak_gbs() if HAVE_TRITON else measured_peak_gbs_torch()
    print(f"measured peak {peak:.1f} GB/s ({'triton' if HAVE_TRITON else 'torch'} probe), KV budget "
          f"{budget.kv_budget_bytes / 2**30:.2f} GiB, baselines: {impls}\n")

    cols = "  ".join(f"{n:>22}" for n in impls)
    print(f"{'b':>3} {'ctx':>7} | {'ours us':>9}  {cols}")
    print("-" * (24 + 24 * len(impls)))
    if not HAVE_TRITON:
        print("(our kernel is n/a here: Triton needs a host C compiler, which this "
              "WSL image lacks -- compare the GB/s column against the Windows tables)")

    rows = []
    for seqlen in [int(x) for x in args.seqlens.split(",") if x]:
        for batch in [int(x) for x in args.batches.split(",") if x]:
            fits, need = budget.fits(shape, batch, seqlen, "fp16", args.block_size)
            if not fits:
                print(f"{batch:3d} {seqlen:7d} | SKIP ({need / 2**30:.2f} GiB)")
                continue
            try:
                r = _run_point(shape, batch, seqlen, args.block_size, sm, peak, impls)
                rows.append(r)
                cells = []
                for n in impls:
                    if f"{n}_ms" in r:
                        cells.append(f"{r[f'{n}_ms']*1e3:8.0f}us {r[f'{n}_gbs']:6.1f}GB/s "
                                     f"{r[f'{n}_pct_peak']:5.1f}%")
                    else:
                        cells.append(f"{'FAIL':>22}")
                ours = f"{r['ours_ms']*1e3:9.0f}" if "ours_ms" in r else f"{'n/a':>9}"
                print(f"{batch:3d} {seqlen:7d} | {ours}  " + "  ".join(cells), flush=True)
            except Exception as exc:
                print(f"{batch:3d} {seqlen:7d} | FAIL {type(exc).__name__}: {str(exc)[:80]}")
            finally:
                gc.collect()
                torch.cuda.empty_cache()

    out_p = pathlib.Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(
        {"gpu": cfg.gpu_info(), "peak_read_gbs_measured": peak,
         "baselines": impls, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out_p}")


if __name__ == "__main__":
    main()

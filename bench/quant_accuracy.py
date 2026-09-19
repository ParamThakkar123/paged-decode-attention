"""How much does a quantized KV cache move the attention output?

    python -m bench.quant_accuracy --out results/quant_accuracy.json

This is the number that decides whether a quantized cache is usable, and it is
deliberately separate from the kernel correctness tests: those compare a kernel
against an fp32 reference reading *the same* quantized cache, so quantization
error cancels and what is measured is the kernel. Here the fp16 cache is the
reference and the quantized caches are the thing under test.

Both caches are built from the same RNG seed, so they hold the same underlying
values and the only difference is the storage format.

Reported as relative error against the fp16 result: max over all elements (the
worst case a token can see) and RMS (what the distribution actually looks like),
because a single outlier and a systematic shift mean very different things for
generation quality.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pagedattn import cache as cache_mod  # noqa: E402
from pagedattn import config as cfg  # noqa: E402
from pagedattn.triton_decode import paged_decode_triton  # noqa: E402


def measure(shape, batch, seqlen, kv_dtype, seed=11) -> dict:
    torch.manual_seed(seed)
    q = torch.randn(batch, shape.num_q_heads, shape.head_dim,
                    dtype=torch.float16, device="cuda")

    c_ref = cache_mod.allocate(shape, batch, seqlen, kv_dtype="fp16", seed=seed)
    ref = paged_decode_triton(q, c_ref).float()
    del c_ref
    torch.cuda.empty_cache()

    c_q = cache_mod.allocate(shape, batch, seqlen, kv_dtype=kv_dtype, seed=seed)
    got = paged_decode_triton(q, c_q).float()
    bytes_per_token = cfg.kv_bytes_per_token(shape, kv_dtype)  # type: ignore[arg-type]
    del c_q
    torch.cuda.empty_cache()

    err = (got - ref).abs()
    scale = ref.abs().max().clamp_min(1e-6)
    rms = (err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    # Cosine similarity per (batch, head): attention output direction is what
    # the next projection actually consumes.
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1).min().item()
    return {
        "kv_dtype": kv_dtype,
        "batch": batch,
        "seqlen": seqlen,
        "bytes_per_token_per_layer": bytes_per_token,
        "max_rel_err": (err.max() / scale).item(),
        "rms_rel_err": rms,
        "min_cosine_similarity": cos,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/quant_accuracy.json")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seqlens", default="2048,8192")
    args = ap.parse_args()

    shape = cfg.LLAMA3_8B
    rows = []
    hdr = (f"{'kv dtype':12s} {'ctx':>7s} {'B/token':>9s} {'max rel':>10s} "
           f"{'rms rel':>10s} {'min cos':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for seqlen in [int(x) for x in args.seqlens.split(",") if x]:
        for kv_dtype in ("fp8_e5m2", "int8"):
            r = measure(shape, args.batch, seqlen, kv_dtype)
            rows.append(r)
            print(f"{r['kv_dtype']:12s} {seqlen:7d} {r['bytes_per_token_per_layer']:9.0f} "
                  f"{r['max_rel_err']:9.4f}  {r['rms_rel_err']:9.4f}  "
                  f"{r['min_cosine_similarity']:9.5f}", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gpu": cfg.gpu_info(), "rows": rows}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

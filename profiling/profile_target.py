"""A single, isolated kernel launch for Nsight Compute / Nsight Systems to attach to.

Nsight Compute replays each kernel many times, so the target has to be small and
deterministic: one allocation, a fixed number of launches, nothing else on the
stream. Run it under `ncu`/`nsys` via the scripts next to this file, or directly
to sanity-check the configuration first.

    python profiling/profile_target.py --impl triton --batch 1 --seqlen 16384
    python profiling/profile_target.py --impl cuda --batch 32 --seqlen 4096 --iters 5
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pagedattn import cuda_decode  # noqa: E402
from pagedattn import cache as cache_mod  # noqa: E402
from pagedattn import config as cfg  # noqa: E402
from pagedattn.triton_decode import paged_decode_triton, pick_num_splits  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="triton",
                    choices=["triton", "triton_nosplit", "triton_pertoken_bt",
                             "cuda", "cuda_nosplit"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seqlen", type=int, default=16384)
    ap.add_argument("--kv-dtype", default="fp16")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--splits", type=int, default=0, help="0 = use the heuristic")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3,
                    help="launches before the region of interest; Triton needs at "
                         "least one to JIT-compile, or the profile captures nvcc")
    args = ap.parse_args()

    shape = cfg.LLAMA3_8B
    sm = torch.cuda.get_device_properties(0).multi_processor_count

    c = cache_mod.allocate(shape, args.batch, args.seqlen, block_size=args.block_size,
                           kv_dtype=args.kv_dtype, shuffle=True, seed=0)
    q = torch.randn(args.batch, shape.num_q_heads, shape.head_dim,
                    dtype=torch.float16, device="cuda")

    if args.impl == "triton":
        ns = args.splits or pick_num_splits(args.batch, shape.num_kv_heads, args.seqlen, sm, 64)
        fn = lambda: paged_decode_triton(q, c, num_splits=ns, per_page_bt=True)  # noqa: E731
    elif args.impl == "triton_nosplit":
        ns = 1
        fn = lambda: paged_decode_triton(q, c, num_splits=1, per_page_bt=True)  # noqa: E731
    elif args.impl == "triton_pertoken_bt":
        ns = args.splits or pick_num_splits(args.batch, shape.num_kv_heads, args.seqlen, sm, 64)
        fn = lambda: paged_decode_triton(q, c, num_splits=ns, per_page_bt=False)  # noqa: E731
    elif args.impl == "cuda":
        ns = args.splits or pick_num_splits(args.batch, shape.num_kv_heads, args.seqlen, sm, 256)
        fn = lambda: cuda_decode.paged_decode_cuda(q, c, num_splits=ns)  # noqa: E731
    else:
        ns = 1
        fn = lambda: cuda_decode.paged_decode_cuda(q, c, num_splits=1)  # noqa: E731

    kv_mb = c.bytes_read_per_decode_step() / 2**20
    print(f"impl={args.impl} batch={args.batch} seqlen={args.seqlen} "
          f"kv_dtype={args.kv_dtype} splits={ns} ctas={args.batch*shape.num_kv_heads*ns} "
          f"sm_count={sm} kv_read={kv_mb:.1f} MiB", flush=True)

    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push(f"{args.impl}_b{args.batch}_s{args.seqlen}")
    for _ in range(args.iters):
        fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    print("done", flush=True)


if __name__ == "__main__":
    main()

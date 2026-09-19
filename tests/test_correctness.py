"""Correctness of every kernel against an fp32 reference.

Run:  python -m pytest tests -q          (or: python tests/test_correctness.py)

Kernel and reference read the *same* cache, so quantization error cancels and
only fp16 accumulation noise is left -- 1e-2 relative is loose for that and
still catches a wrong block-table index or a botched softmax rescale.
Quantization error against an fp16 cache is measured separately, in
`test_quantization_accuracy`.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pagedattn import LLAMA3_8B, MHA_DEBUG, allocate, cuda_decode, reference  # noqa: E402
from pagedattn.cache import gather_contiguous  # noqa: E402
from pagedattn.triton_decode import paged_decode_triton  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

REL_TOL = 1e-2

# Sizes chosen to hit the awkward cases on purpose:
#   64   -> exactly one BLOCK_N tile
#   65   -> one token past a tile, tail mask must work
#   300  -> not a multiple of the page size
#   1024 -> enough tokens for a meaningful split
#   4097 -> odd length with many splits
SHAPES = [(2, 64), (2, 65), (3, 300), (1, 1024), (4, 1023), (2, 4097)]


def _qkv(batch, seqlen, kv_dtype="fp16", block_size=16, shape=LLAMA3_8B, seed=0):
    torch.manual_seed(seed)
    c = allocate(shape, batch, seqlen, block_size=block_size,
                 kv_dtype=kv_dtype, seed=seed)  # type: ignore[arg-type]
    q = torch.randn(batch, shape.num_q_heads, shape.head_dim,
                    dtype=torch.float16, device="cuda")
    return q, c


def _rel(got, ref):
    return (got.float() - ref.float()).abs().max().item() / max(
        ref.float().abs().max().item(), 1e-6
    )


@pytest.mark.parametrize("batch,seqlen", SHAPES)
@pytest.mark.parametrize("num_splits", [1, 4])
@pytest.mark.parametrize("per_page_bt", [True, False])
def test_triton_matches_reference(batch, seqlen, num_splits, per_page_bt):
    q, c = _qkv(batch, seqlen)
    ref = reference.reference_decode(q, c)
    got = paged_decode_triton(q, c, num_splits=num_splits, per_page_bt=per_page_bt)
    assert _rel(got, ref) < REL_TOL


@pytest.mark.parametrize("kv_dtype", ["fp16", "fp8_e5m2", "int8"])
@pytest.mark.parametrize("num_splits", [1, 8])
def test_triton_quantized_kv(kv_dtype, num_splits):
    q, c = _qkv(2, 2048, kv_dtype=kv_dtype)
    ref = reference.reference_decode(q, c)
    got = paged_decode_triton(q, c, num_splits=num_splits)
    assert _rel(got, ref) < REL_TOL


@pytest.mark.parametrize("block_size", [16, 32, 64])
def test_triton_page_sizes(block_size):
    q, c = _qkv(3, 1000, block_size=block_size)
    ref = reference.reference_decode(q, c)
    got = paged_decode_triton(q, c, block_n=max(64, block_size))
    assert _rel(got, ref) < REL_TOL


def test_ragged_batch():
    """Mixed sequence lengths in one batch -- what a real server always has."""
    torch.manual_seed(0)
    lens = [7, 1, 4096, 513, 16, 2047, 64, 129]
    c = allocate(LLAMA3_8B, len(lens), lens, seed=3)
    q = torch.randn(len(lens), LLAMA3_8B.num_q_heads, LLAMA3_8B.head_dim,
                    dtype=torch.float16, device="cuda")
    ref = reference.reference_decode(q, c)
    for ns in (1, 4):
        assert _rel(paged_decode_triton(q, c, num_splits=ns), ref) < REL_TOL
    if cuda_decode.is_available():
        for ns in (1, 4):
            assert _rel(cuda_decode.paged_decode_cuda(q, c, num_splits=ns), ref) < REL_TOL


def test_mha_group_one():
    """group=1 exercises the GROUP_PAD masking path that Llama-3 never hits."""
    q, c = _qkv(2, 512, shape=MHA_DEBUG)
    ref = reference.reference_decode(q, c)
    assert _rel(paged_decode_triton(q, c), ref) < REL_TOL


def test_fragmented_vs_sequential_blocks_agree():
    """A shuffled block table must not change the answer, only the speed."""
    torch.manual_seed(0)
    q = torch.randn(2, LLAMA3_8B.num_q_heads, LLAMA3_8B.head_dim,
                    dtype=torch.float16, device="cuda")
    outs = []
    for shuffle in (False, True):
        c = allocate(LLAMA3_8B, 2, 1024, shuffle=shuffle, seed=5)
        outs.append((paged_decode_triton(q, c), reference.reference_decode(q, c)))
    for got, ref in outs:
        assert _rel(got, ref) < REL_TOL


# ---------------------------------------------------------------------------
# CUDA kernel
# ---------------------------------------------------------------------------

cuda_only = pytest.mark.skipif(
    not cuda_decode.is_available(), reason=f"CUDA ext unavailable: {cuda_decode.load_error()}"
)


@cuda_only
@pytest.mark.parametrize("batch,seqlen", SHAPES)
@pytest.mark.parametrize("num_splits", [1, 4])
def test_cuda_matches_reference(batch, seqlen, num_splits):
    q, c = _qkv(batch, seqlen)
    ref = reference.reference_decode(q, c)
    got = cuda_decode.paged_decode_cuda(q, c, num_splits=num_splits)
    assert _rel(got, ref) < REL_TOL


@cuda_only
@pytest.mark.parametrize("kv_dtype", ["fp16", "fp8_e5m2", "int8"])
@pytest.mark.parametrize("num_splits", [1, 8])
def test_cuda_quantized_kv(kv_dtype, num_splits):
    q, c = _qkv(2, 2048, kv_dtype=kv_dtype)
    ref = reference.reference_decode(q, c)
    got = cuda_decode.paged_decode_cuda(q, c, num_splits=num_splits)
    assert _rel(got, ref) < REL_TOL


@cuda_only
@pytest.mark.parametrize("block_size", [16, 32, 64])
def test_cuda_page_sizes(block_size):
    q, c = _qkv(3, 1000, block_size=block_size)
    ref = reference.reference_decode(q, c)
    assert _rel(cuda_decode.paged_decode_cuda(q, c), ref) < REL_TOL


@cuda_only
def test_triton_and_cuda_agree():
    q, c = _qkv(4, 3000)
    a = paged_decode_triton(q, c, num_splits=4)
    b = cuda_decode.paged_decode_cuda(q, c, num_splits=4)
    assert _rel(a, b) < REL_TOL


# ---------------------------------------------------------------------------
# baselines and quantization quality
# ---------------------------------------------------------------------------


def test_sdpa_math_baseline_matches_reference():
    q, c = _qkv(2, 512)
    ref = reference.reference_decode(q, c)
    k, v = gather_contiguous(c)
    assert _rel(reference.sdpa_decode(q, k, v, c.seq_lens, "math"), ref) < REL_TOL


@pytest.mark.parametrize("kv_dtype", ["fp8_e5m2", "int8"])
def test_quantization_accuracy(kv_dtype):
    """Quantized KV vs an fp16 cache holding the same underlying values.

    The serving-relevant number: how far the output moves when the cache is
    halved. INT8 with per-(token,head) scales lands near fp16; e5m2 keeps 2
    mantissa bits and is visibly worse.
    """
    batch, seqlen = 2, 2048
    q, c_fp16 = _qkv(batch, seqlen, kv_dtype="fp16", seed=11)
    _, c_q = _qkv(batch, seqlen, kv_dtype=kv_dtype, seed=11)

    ref = reference.reference_decode(q, c_fp16)
    got = paged_decode_triton(q, c_q)
    rel = _rel(got, ref)
    # Just above the measured values (int8 ~0.4%, e5m2 ~8%), so a dequant
    # regression fails rather than only a totally broken path.
    # `python -m bench.quant_accuracy` reports RMS and cosine too.
    ceiling = {"int8": 0.02, "fp8_e5m2": 0.15}[kv_dtype]
    assert rel < ceiling, f"{kv_dtype} relative error {rel:.4f} exceeded {ceiling}"
    print(f"\n{kv_dtype}: max relative output error vs fp16 cache = {rel:.4f}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))


# ---------------------------------------------------------------------------
# shapes beyond the Llama-3-8B benchmark shape
# ---------------------------------------------------------------------------

# (num_q_heads, num_kv_heads, head_dim) for models that actually exist:
#   32/8/128  Llama-3-8B          (the benchmark shape)
#   64/8/128  Llama-3-70B
#   32/8/64   Llama-3.2-1B
#   14/2/64   Qwen2.5-0.5B        (the model the vLLM integration runs)
#   16/2/64   a group-8 / head_dim-64 case
OTHER_SHAPES = [(32, 8, 128), (64, 8, 128), (32, 8, 64), (14, 2, 64), (16, 2, 64)]


@pytest.mark.parametrize("hq,hkv,d", OTHER_SHAPES)
@pytest.mark.parametrize("num_splits", [1, 4])
def test_triton_other_model_shapes(hq, hkv, d, num_splits):
    """Takes group and head_dim as constexpr, so it should compile for any
    real model shape, not just the benchmark one."""
    shape = MHA_DEBUG.__class__(hq, hkv, d, name=f"{hq}x{hkv}x{d}")
    q, c = _qkv(3, 777, shape=shape, seed=7)
    ref = reference.reference_decode(q, c)
    got = paged_decode_triton(q, c, num_splits=num_splits)
    assert _rel(got, ref) < REL_TOL


@cuda_only
@pytest.mark.parametrize("hq,hkv,d", OTHER_SHAPES)
def test_cuda_other_model_shapes(hq, hkv, d):
    """Instantiated per (head_dim, group): supported pairs must work, and
    unsupported ones must raise rather than run a wrong specialization."""
    shape = MHA_DEBUG.__class__(hq, hkv, d, name=f"{hq}x{hkv}x{d}")
    q, c = _qkv(3, 777, shape=shape, seed=7)
    ok, _ = cuda_decode.supports_shape(d, hq // hkv)
    if not ok:
        with pytest.raises(ValueError):
            cuda_decode.paged_decode_cuda(q, c)
        return
    ref = reference.reference_decode(q, c)
    assert _rel(cuda_decode.paged_decode_cuda(q, c, num_splits=1), ref) < REL_TOL
    assert _rel(cuda_decode.paged_decode_cuda(q, c, num_splits=4), ref) < REL_TOL


@cuda_only
def test_cuda_rejects_uncompiled_shape():
    """head_dim 256 is not instantiated; the error must name the shape."""
    ok, why = cuda_decode.supports_shape(256, 4)
    assert not ok and "256" in why and "Triton" in why

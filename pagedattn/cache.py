"""Paged KV cache: allocation, block tables, and the quantized variants.

Layout is vLLM-v1 / FlashInfer "NHD":

    k_cache, v_cache : [num_blocks, block_size, num_kv_heads, head_dim]

head_dim varies fastest, so a (block, token, head) row is contiguous (256 B in
fp16) and the kernels issue coalesced 128-bit loads.

`allocate(shuffle=...)` picks scattered or ascending physical blocks. Real
servers are fragmented, so scattered is the default; sequential is kept only as
a locality upper bound.
"""

from __future__ import annotations

import dataclasses

import torch

from .config import KVDType, ModelShape, torch_dtype

# ----------------------------------------------------------------------------
# quantization helpers
# ----------------------------------------------------------------------------

# e5m2 shares fp16's 5-bit exponent, so the conversion is pure mantissa
# truncation -- no scale factor, and Ampere converts back with a shift.
FP8_E5M2_MAX = 57344.0
INT8_MAX = 127.0


def quantize_kv(x: torch.Tensor, kv_dtype: KVDType) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Quantize a [..., num_kv_heads, head_dim] fp16 tensor to the cache dtype.

    Returns (stored, scale). `scale` is None for the non-scaled dtypes and
    [..., num_kv_heads] fp16 for int8 (one scale per token per head).
    """
    if kv_dtype in ("fp16", "bf16"):
        return x.to(torch_dtype(kv_dtype)), None
    if kv_dtype == "fp8_e5m2":
        return x.to(torch.float8_e5m2), None
    if kv_dtype == "int8":
        amax = x.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-6)
        scale = (amax / INT8_MAX).to(torch.float16)
        q = (x.float() / scale.float()).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
        return q, scale.squeeze(-1)
    raise ValueError(f"unknown kv_dtype {kv_dtype}")


def dequantize_kv(
    q: torch.Tensor, scale: torch.Tensor | None, kv_dtype: KVDType
) -> torch.Tensor:
    """Inverse of `quantize_kv`, in fp16. Used by the reference implementation."""
    if kv_dtype in ("fp16", "bf16"):
        return q.to(torch.float16)
    if kv_dtype == "fp8_e5m2":
        return q.to(torch.float16)
    if kv_dtype == "int8":
        assert scale is not None
        return (q.to(torch.float16) * scale.unsqueeze(-1)).to(torch.float16)
    raise ValueError(f"unknown kv_dtype {kv_dtype}")


# ----------------------------------------------------------------------------
# the cache
# ----------------------------------------------------------------------------


@dataclasses.dataclass
class PagedKVCache:
    shape: ModelShape
    block_size: int
    kv_dtype: KVDType
    k_cache: torch.Tensor  # [num_blocks, block_size, H_kv, D]
    v_cache: torch.Tensor
    k_scale: torch.Tensor | None  # [num_blocks, block_size, H_kv] fp16, int8 only
    v_scale: torch.Tensor | None
    block_table: torch.Tensor  # [B, max_blocks] int32
    seq_lens: torch.Tensor  # [B] int32

    @property
    def batch(self) -> int:
        return int(self.block_table.shape[0])

    @property
    def num_blocks(self) -> int:
        return int(self.k_cache.shape[0])

    def bytes_read_per_decode_step(self) -> int:
        """KV bytes a perfect decode kernel must read.

        The denominator of every achieved-bandwidth number here. Counts only KV
        inside the sequence, since a correct kernel masks the page-padding tail.
        """
        n_tok = int(self.seq_lens.sum().item())
        h, d = self.shape.num_kv_heads, self.shape.head_dim
        elem = self.k_cache.element_size()
        total = 2 * n_tok * h * d * elem
        if self.kv_dtype == "int8":
            total += 2 * n_tok * h * 2  # fp16 scales
        return total


def allocate(
    shape: ModelShape,
    batch: int,
    seq_lens: torch.Tensor | list[int] | int,
    block_size: int = 16,
    kv_dtype: KVDType = "fp16",
    device: str = "cuda",
    shuffle: bool = True,
    seed: int = 0,
    slack_blocks: int = 0,
) -> PagedKVCache:
    """Allocate a paged cache and fill it with a realistic random KV state."""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    if isinstance(seq_lens, int):
        seq_lens_t = torch.full((batch,), seq_lens, dtype=torch.int32)
    elif isinstance(seq_lens, list):
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    else:
        seq_lens_t = seq_lens.to(torch.int32).cpu()
    assert seq_lens_t.numel() == batch

    blocks_per_seq = ((seq_lens_t + block_size - 1) // block_size).tolist()
    max_blocks = max(blocks_per_seq)
    total_blocks = sum(blocks_per_seq) + slack_blocks

    h, d = shape.num_kv_heads, shape.head_dim
    store_dtype = torch_dtype(kv_dtype)

    # Block-wise fill, so the fp16 staging buffer fits a 4 GB card.
    k_cache = torch.empty((total_blocks, block_size, h, d), dtype=store_dtype, device=device)
    v_cache = torch.empty_like(k_cache)
    need_scale = kv_dtype == "int8"
    k_scale = (
        torch.empty((total_blocks, block_size, h), dtype=torch.float16, device=device)
        if need_scale
        else None
    )
    v_scale = torch.empty_like(k_scale) if k_scale is not None else None

    chunk = max(1, min(total_blocks, (32 << 20) // (block_size * h * d * 2)))
    gpu_gen = torch.Generator(device=device).manual_seed(seed)
    for start in range(0, total_blocks, chunk):
        stop = min(start + chunk, total_blocks)
        for src_cache, src_scale in ((k_cache, k_scale), (v_cache, v_scale)):
            raw = torch.randn(
                (stop - start, block_size, h, d),
                dtype=torch.float16,
                device=device,
                generator=gpu_gen,
            )
            q, s = quantize_kv(raw, kv_dtype)
            src_cache[start:stop] = q
            if s is not None:
                assert src_scale is not None
                src_scale[start:stop] = s
            del raw, q, s

    perm = (
        torch.randperm(total_blocks, generator=gen)[: sum(blocks_per_seq)]
        if shuffle
        else torch.arange(sum(blocks_per_seq))
    )
    block_table = torch.zeros((batch, max_blocks), dtype=torch.int32)
    cursor = 0
    for b, nb in enumerate(blocks_per_seq):
        block_table[b, :nb] = perm[cursor : cursor + nb].to(torch.int32)
        cursor += nb

    return PagedKVCache(
        shape=shape,
        block_size=block_size,
        kv_dtype=kv_dtype,
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
        block_table=block_table.to(device),
        seq_lens=seq_lens_t.to(device),
    )


def gather_contiguous(cache: PagedKVCache) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize the paged cache into dense [B, H_kv, S_max, D] fp16 tensors.

    The dense baselines have no notion of pages, so somebody pays this gather.
    `bench_decode.py` times it separately and reports both gather-inclusive and
    gather-exclusive numbers -- which is fair depends on whether the argument is
    about kernel quality or serving cost.
    """
    b = cache.batch
    s_max = int(cache.seq_lens.max().item())
    dev = cache.k_cache.device
    bs = cache.block_size

    pos = torch.arange(s_max, device=dev)
    blk = pos // bs
    off = pos % bs
    nblk = cache.block_table.shape[1]
    phys = cache.block_table[:, blk.clamp(max=nblk - 1)].long()  # [B, S]

    out = []
    for cache_t, scale_t in ((cache.k_cache, cache.k_scale), (cache.v_cache, cache.v_scale)):
        flat = cache_t[phys, off.unsqueeze(0).expand(b, -1)]  # [B, S, H, D]
        if scale_t is not None:
            s = scale_t[phys, off.unsqueeze(0).expand(b, -1)]  # [B, S, H]
            flat = dequantize_kv(flat, s, cache.kv_dtype)
        else:
            flat = dequantize_kv(flat, None, cache.kv_dtype)
        out.append(flat.permute(0, 2, 1, 3).contiguous())  # [B, H, S, D]
    return out[0], out[1]

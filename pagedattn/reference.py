"""Reference decode attention + the PyTorch/FA2 baselines.

`reference_decode` is deliberately slow and obvious: fp32 math, explicit gather,
no fusion. It is the ground truth that `tests/test_correctness.py` compares every
kernel against.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .cache import PagedKVCache, gather_contiguous

# PyTorch ships the cuDNN SDPA backend *runtime-disabled* on this build, so it
# reports "cuDNN attention has been runtime disabled" and silently never runs.
# Turning it on is worth the line: unlike the mem-efficient backend it accepts
# GQA shapes directly via `enable_gqa`, so it is the only fused baseline here
# that does not need the KV heads physically expanded 8 -> 32 first.
if torch.cuda.is_available() and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(True)

_BACKENDS = {
    "math": "MATH",
    "mem_efficient": "EFFICIENT_ATTENTION",
    "flash": "FLASH_ATTENTION",
    "cudnn": "CUDNN_ATTENTION",
}


def reference_decode(q: torch.Tensor, cache: PagedKVCache) -> torch.Tensor:
    """fp32 paged GQA decode attention. q: [B, H_q, D] -> out: [B, H_q, D]."""
    b, hq, d = q.shape
    hkv = cache.shape.num_kv_heads
    group = hq // hkv
    k, v = gather_contiguous(cache)  # [B, H_kv, S, D] fp16
    k = k.float()
    v = v.float()
    qf = q.float()

    s_max = k.shape[2]
    pos = torch.arange(s_max, device=q.device)
    valid = pos.unsqueeze(0) < cache.seq_lens.unsqueeze(1)  # [B, S]

    out = torch.empty((b, hq, d), dtype=torch.float32, device=q.device)
    for h in range(hq):
        kvh = h // group
        logits = torch.einsum("bd,bsd->bs", qf[:, h], k[:, kvh]) * cache.shape.softmax_scale
        logits = logits.masked_fill(~valid, float("-inf"))
        p = torch.softmax(logits, dim=-1)
        out[:, h] = torch.einsum("bs,bsd->bd", p, v[:, kvh])
    return out.to(q.dtype)


# ----------------------------------------------------------------------------
# baselines
# ----------------------------------------------------------------------------


def _sdpa(q, k, v, seq_lens, backend: str | None) -> torch.Tensor:
    """q: [B,Hq,1,D], k/v: [B,Hkv,S,D] -> [B,Hq,D]."""
    b, hq, _, d = q.shape
    s = k.shape[2]
    pos = torch.arange(s, device=q.device)
    # [B, 1, 1, S] boolean mask; True = attend.
    mask = (pos.view(1, 1, 1, -1) < seq_lens.view(-1, 1, 1, 1)).expand(b, 1, 1, s)

    kwargs = dict(attn_mask=mask, enable_gqa=(hq != k.shape[1]))
    if backend is None:
        o = F.scaled_dot_product_attention(q, k, v, **kwargs)
    else:
        sel = getattr(torch.nn.attention.SDPBackend, _BACKENDS[backend])
        with torch.nn.attention.sdpa_kernel(sel):
            o = F.scaled_dot_product_attention(q, k, v, **kwargs)
    return o.squeeze(2)


def sdpa_gqa_uniform(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    backend: str = "cudnn",
) -> torch.Tensor:
    """Fused GQA decode with no attention mask and no KV-head expansion.

    The cuDNN backend rejects an arbitrary `attn_mask`, so this path drops it --
    which is only correct when every sequence in the batch has the same length,
    i.e. exactly the sweep's configuration. `enable_gqa=True` means the KV stays
    at 8 heads, so unlike the mem-efficient baseline this one does not pay 4x
    the KV memory to be callable at all.

    q: [B, Hq, D], k/v: [B, Hkv, S, D] -> [B, Hq, D].
    """
    sel = getattr(torch.nn.attention.SDPBackend, _BACKENDS[backend])
    with torch.nn.attention.sdpa_kernel(sel):
        o = F.scaled_dot_product_attention(
            q.unsqueeze(2), k, v, attn_mask=None, is_causal=False,
            enable_gqa=(q.shape[1] != k.shape[1]),
        )
    return o.squeeze(2)


def sdpa_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: torch.Tensor,
    backend: str | None = None,
) -> torch.Tensor:
    """PyTorch SDPA baseline on already-gathered contiguous KV."""
    return _sdpa(q.unsqueeze(2), k, v, seq_lens, backend)


def sdpa_flash_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """FlashAttention-2 path.

    PyTorch's FLASH_ATTENTION SDPA backend *is* FlashAttention-2 (the upstream
    kernels are vendored into ATen), which is how we get an FA2 baseline without
    the flash-attn package -- it has no Windows wheels. FA2 cannot take an
    arbitrary attn_mask, so ragged batches are run with is_causal=False and a
    right-padded KV; we therefore only use this path when all sequences in the
    batch have equal length, which is the case for every sweep point.
    """
    assert int(seq_lens.min()) == int(seq_lens.max()), "FA2 path needs uniform seq_lens"
    hq, hkv = q.shape[1], k.shape[1]
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        o = F.scaled_dot_product_attention(
            q.unsqueeze(2), k, v, attn_mask=None, is_causal=False, enable_gqa=(hq != hkv)
        )
    return o.squeeze(2)

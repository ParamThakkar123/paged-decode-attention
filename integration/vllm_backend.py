"""A vLLM V1 attention backend that routes **decode** to our Triton kernel.

Enable it with the helper in `pagedattn_plugin.py`, or by hand:

    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    PYTHONPATH=/path/to/inference_benchmark \
    python -c "import integration.vllm_backend as b; b.install(); ..."

Design: **subclass vLLM's own `TritonAttentionBackend` and override only
`forward`.** Building V1 attention metadata correctly (query_start_loc,
slot_mapping, block tables, cascade/local-attention variants, CUDA-graph capture
paths) is the fiddly, version-sensitive part, and reimplementing it would be
both more code and more ways to be subtly wrong. Everything except the decode
call is inherited.

What we intercept, and what we hand back:

  pure decode  (`max_query_len == 1`)  -> our kernel
  anything else                        -> `super().forward()`

"Anything else" is not a corner case to apologize for -- it is prefill, mixed
prefill+decode batches, fp8 KV caches, sliding-window and ALiBi layers, and
soft-capped logits. Our kernel is a single-query decode kernel with no causal
mask across a query tile and no positional-bias support; pretending otherwise
would produce wrong numbers rather than an error. The delegation is the design,
not a gap.

Layout note: vLLM allocates the V1 KV cache as
`[2, num_blocks, block_size, num_kv_heads, head_size]`, so `kv_cache.unbind(0)`
yields exactly the NHD tensors our kernel already expects -- no conversion, no
copy. That is the payoff for matching this layout in `pagedattn/cache.py`.
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pagedattn.cache import PagedKVCache  # noqa: E402
from pagedattn.config import ModelShape  # noqa: E402
from pagedattn.triton_decode import paged_decode_triton, pick_num_splits  # noqa: E402

# Importing this module must not require vLLM: the Windows side imports it for
# lint and shape-gate tests. The classes below only exist where vLLM does.
try:
    from vllm.v1.attention.backends.triton_attn import (  # noqa: E402
        TritonAttentionBackend, TritonAttentionImpl)
    HAVE_VLLM = True
except Exception:  # pragma: no cover - depends on the environment
    TritonAttentionBackend = object  # type: ignore[assignment,misc]
    TritonAttentionImpl = object  # type: ignore[assignment,misc]
    HAVE_VLLM = False


def supports_shape(num_q_heads: int, num_kv_heads: int, head_dim: int) -> tuple[bool, str]:
    """Whether our decode kernel can serve this attention shape."""
    if num_q_heads % num_kv_heads != 0:
        return False, f"num_q_heads {num_q_heads} not divisible by num_kv_heads {num_kv_heads}"
    if head_dim not in (32, 64, 128, 256):
        return False, f"head_dim {head_dim} not in (32, 64, 128, 256)"
    return True, ""


# Counters, so a run can prove the kernel was actually used rather than silently
# delegated. `bench_vllm.py` prints them.
STATS = {"decode_calls": 0, "decode_tokens": 0, "delegated_calls": 0}


def _reason_to_delegate(impl, attn_metadata, kv_cache, output, output_scale) -> str:
    if attn_metadata is None:
        return "profiling run (no metadata)"
    if output is None:
        return "no output buffer"
    if output_scale is not None:
        return "fused output quantization"
    if kv_cache is None or kv_cache.numel() == 0:
        return "empty kv cache"
    if getattr(impl, "kv_cache_dtype", "auto").startswith("fp8"):
        return "fp8 kv cache"
    if getattr(impl, "alibi_slopes", None) is not None:
        return "alibi slopes"
    if getattr(impl, "logits_soft_cap", 0):
        return "logits soft cap"
    if tuple(getattr(impl, "sliding_window", (-1, -1))) != (-1, -1):
        return "sliding window"
    if getattr(attn_metadata, "local_attn_metadata", None) is not None:
        return "local (iRoPE) attention"
    if getattr(attn_metadata, "use_cascade", False):
        return "cascade attention"
    if getattr(attn_metadata, "max_query_len", 2) != 1:
        return "not a pure-decode batch"
    return ""


if HAVE_VLLM:

    class PagedAttnTritonImpl(TritonAttentionImpl):  # type: ignore[misc]
        """Decode goes to our kernel; everything else to vLLM's."""

        def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                    output=None, output_scale=None):
            why = _reason_to_delegate(self, attn_metadata, kv_cache, output, output_scale)
            if why:
                STATS["delegated_calls"] += 1
                return super().forward(layer, query, key, value, kv_cache,
                                       attn_metadata, output, output_scale)

            ok, shape_why = supports_shape(self.num_heads, self.num_kv_heads,
                                           self.head_size)
            if not ok:
                STATS["delegated_calls"] += 1
                return super().forward(layer, query, key, value, kv_cache,
                                       attn_metadata, output, output_scale)

            key_cache, value_cache = kv_cache.unbind(0)

            # Write this step's K/V into the paged cache. vLLM's own impl does
            # this too; we are replacing only the attention that follows it.
            if self.kv_sharing_target_layer_name is None:
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    key, value, key_cache, value_cache,
                    attn_metadata.slot_mapping, self.kv_cache_dtype,
                    layer._k_scale, layer._v_scale,
                )

            n = attn_metadata.num_actual_tokens
            q = query[:n]                      # [n, num_heads, head_size]
            out = output[:n]

            cache = PagedKVCache(
                shape=ModelShape(self.num_heads, self.num_kv_heads,
                                 self.head_size, name="vllm"),
                block_size=int(key_cache.shape[1]),
                kv_dtype="bf16" if key_cache.dtype == torch.bfloat16 else "fp16",
                k_cache=key_cache, v_cache=value_cache,
                k_scale=None, v_scale=None,
                block_table=attn_metadata.block_table[:n],
                seq_lens=attn_metadata.seq_lens[:n],
            )

            # max_seq_len is already a Python int on the metadata, so choosing a
            # split count costs no device sync -- which is the whole point of
            # README section 5.1.
            splits = pick_num_splits(n, self.num_kv_heads,
                                     int(attn_metadata.max_seq_len),
                                     _sm_count(query.device), 64)

            paged_decode_triton(q, cache, out=out, num_splits=splits,
                                sm_scale=self.scale)

            STATS["decode_calls"] += 1
            STATS["decode_tokens"] += int(n)
            return output

    class PagedAttnTritonBackend(TritonAttentionBackend):  # type: ignore[misc]
        """Same metadata and KV-cache layout as vLLM's Triton backend."""

        @staticmethod
        def get_name() -> str:
            return "PAGEDATTN_TRITON"

        @staticmethod
        def get_impl_cls():
            return PagedAttnTritonImpl

else:  # pragma: no cover - lint/import path without vLLM
    PagedAttnTritonImpl = None  # type: ignore[assignment]
    PagedAttnTritonBackend = None  # type: ignore[assignment]


_SM_COUNT: int | None = None


def _sm_count(device) -> int:
    global _SM_COUNT
    if _SM_COUNT is None:
        _SM_COUNT = torch.cuda.get_device_properties(device).multi_processor_count
    return _SM_COUNT


QUALNAME = "integration.vllm_backend.PagedAttnTritonBackend"


def install() -> str:
    """Make vLLM select this backend.

    vLLM resolves a backend by calling `current_platform.get_attn_backend_cls()`,
    which returns a *qualified name string* that `resolve_obj_by_qualname` then
    imports. `VLLM_ATTENTION_BACKEND` only selects among the built-in `_Backend`
    enum members, so an out-of-tree class cannot be chosen through it; patching
    the platform hook is the supported shape of the extension point.

    Returns the qualname that will now be used.
    """
    from vllm.platforms import current_platform

    cls = type(current_platform)
    if getattr(cls, "_pagedattn_patched", False):
        return QUALNAME
    original = cls.get_attn_backend_cls

    def patched(cls_, selected_backend, head_size, dtype, kv_cache_dtype,
                block_size, use_v1, use_mla, *args, **kwargs):
        # Only intercept the plain (non-MLA) V1 path we actually implement.
        if use_v1 and not use_mla:
            return QUALNAME
        return original(selected_backend, head_size, dtype, kv_cache_dtype,
                        block_size, use_v1, use_mla, *args, **kwargs)

    cls.get_attn_backend_cls = classmethod(patched)
    cls._pagedattn_patched = True
    return QUALNAME


def maybe_install_from_env() -> bool:
    """Install if `PAGEDATTN_VLLM_BACKEND=1`. Called by the plugin entry point."""
    if os.environ.get("PAGEDATTN_VLLM_BACKEND", "") not in ("1", "true", "True"):
        return False
    install()
    return True

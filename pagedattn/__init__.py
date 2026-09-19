"""Paged-KV GQA decode attention kernel."""

from .config import LLAMA3_8B, LLAMA3_70B, MHA_DEBUG, ModelShape, VRamBudget, gpu_info
from .cache import PagedKVCache, allocate, gather_contiguous
from .reference import reference_decode, sdpa_decode, sdpa_flash_varlen
from .triton_decode import paged_decode_triton, pick_num_splits

__all__ = [
    "LLAMA3_8B", "LLAMA3_70B", "MHA_DEBUG", "ModelShape", "VRamBudget", "gpu_info",
    "PagedKVCache", "allocate", "gather_contiguous",
    "reference_decode", "sdpa_decode", "sdpa_flash_varlen",
    "paged_decode_triton", "pick_num_splits",
]

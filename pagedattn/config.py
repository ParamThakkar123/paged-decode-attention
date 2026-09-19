"""Shapes, dtypes and the VRAM budget planner.

Everything here is sized for a 4 GB card. The budget planner is the reason the
sweep in `bench/` never OOMs: it computes the exact KV-cache footprint of a
(batch, context) point up front and skips the point if it does not fit, instead
of discovering that halfway through a benchmark run.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import torch

KVDType = Literal["fp16", "bf16", "fp8_e5m2", "int8"]

# Bytes per stored KV element, per cache dtype.
_ELEM_BYTES: dict[KVDType, float] = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8_e5m2": 1.0,
    "int8": 1.0,
}

_TORCH_DTYPE: dict[KVDType, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8_e5m2": torch.float8_e5m2,
    "int8": torch.int8,
}


@dataclasses.dataclass(frozen=True)
class ModelShape:
    """Llama-3-8B attention shape (the default) or anything GQA-shaped."""

    num_q_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    name: str = "llama3-8b"

    @property
    def gqa_group(self) -> int:
        assert self.num_q_heads % self.num_kv_heads == 0, (
            f"num_q_heads={self.num_q_heads} not divisible by "
            f"num_kv_heads={self.num_kv_heads}"
        )
        return self.num_q_heads // self.num_kv_heads

    @property
    def softmax_scale(self) -> float:
        return self.head_dim**-0.5


LLAMA3_8B = ModelShape(32, 8, 128, "llama3-8b")
LLAMA3_70B = ModelShape(64, 8, 128, "llama3-70b")
MHA_DEBUG = ModelShape(8, 8, 128, "mha-debug")  # group=1, exercises the G=1 path


def kv_bytes_per_token(shape: ModelShape, kv_dtype: KVDType) -> float:
    """Bytes of K+V cache for one token of one layer."""
    per = 2 * shape.num_kv_heads * shape.head_dim * _ELEM_BYTES[kv_dtype]
    if kv_dtype == "int8":
        # per-(token, head) fp16 scale for K and V
        per += 2 * shape.num_kv_heads * 2
    return per


def torch_dtype(kv_dtype: KVDType) -> torch.dtype:
    return _TORCH_DTYPE[kv_dtype]


@dataclasses.dataclass
class VRamBudget:
    """Decides which (batch, seqlen) points are runnable on this GPU."""

    total_bytes: int
    # Fraction of total VRAM the KV cache may occupy. The rest pays for the
    # CUDA context (~250 MB), cuBLAS/cuDNN workspaces, Q/O, the split-KV
    # workspace and the SDPA baselines' buffers.
    kv_fraction: float = 0.55

    @classmethod
    def from_device(cls, device: int = 0, kv_fraction: float = 0.55) -> "VRamBudget":
        props = torch.cuda.get_device_properties(device)
        return cls(total_bytes=props.total_memory, kv_fraction=kv_fraction)

    @property
    def kv_budget_bytes(self) -> int:
        return int(self.total_bytes * self.kv_fraction)

    def fits(
        self,
        shape: ModelShape,
        batch: int,
        seqlen: int,
        kv_dtype: KVDType,
        block_size: int,
    ) -> tuple[bool, int]:
        """Return (fits, required_bytes) including page-rounding waste."""
        blocks_per_seq = (seqlen + block_size - 1) // block_size
        padded_tokens = blocks_per_seq * block_size * batch
        need = int(padded_tokens * kv_bytes_per_token(shape, kv_dtype))
        return need <= self.kv_budget_bytes, need


def gpu_info(device: int = 0) -> dict:
    props = torch.cuda.get_device_properties(device)
    return {
        "name": props.name,
        "sm_count": props.multi_processor_count,
        "capability": f"{props.major}.{props.minor}",
        "total_vram_gb": round(props.total_memory / 1024**3, 3),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "driver_cuda": _driver_cuda_version(),
    }


def _driver_cuda_version() -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - diagnostic only
        return "unknown"

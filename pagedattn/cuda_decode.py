"""JIT-compiled CUDA paged decode attention.

The extension is built on first use with `torch.utils.cpp_extension.load` and
cached under `build/`. On Windows with CUDA 12.3 + MSVC 14.4x nvcc refuses the
host compiler by default, hence `-allow-unsupported-compiler`; the generated
code is unaffected (it is a version-table check, not a codegen difference).
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import warnings

import torch

from .cache import PagedKVCache
from .triton_decode import _KV_CODE, _Workspace, pick_num_splits

_HERE = pathlib.Path(__file__).parent
_SRC = _HERE / "csrc" / "paged_decode.cu"
_BUILD = _HERE.parent / "build" / "cuda_ext"

_EXT = None
_LOAD_ERROR: str | None = None
_WS = _Workspace()
# Hoisted: an undefined tensor stands in for the optional arguments, and
# allocating one per call shows up in the profile at short context lengths.
_EMPTY = torch.Tensor()


def _ensure_msvc_env() -> None:
    """Make nvcc able to find MSVC regardless of which shell we were started from.

    nvcc shells out to vcvars64.bat, and that script breaks if Git Bash's
    /usr/bin is on PATH -- its `sort` and `find` shadow the Windows ones and
    vcvars exits with "Could not set up the environment for Microsoft Visual
    Studio". Rather than require a Developer Prompt, we run vcvars ourselves with
    a sanitized PATH and import the resulting variables into this process.
    """
    if os.name != "nt" or "VCToolsInstallDir" in os.environ:
        return

    vswhere = pathlib.Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / (
        r"Microsoft Visual Studio\Installer\vswhere.exe"
    )
    roots: list[pathlib.Path] = []
    if vswhere.exists():
        try:
            res = subprocess.run(
                [str(vswhere), "-latest", "-products", "*",
                 "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "installationPath"],
                capture_output=True, text=True, timeout=60,
            )
            if res.stdout.strip():
                roots.append(pathlib.Path(res.stdout.strip()))
        except Exception:
            pass
    for edition in ("Community", "Professional", "Enterprise", "BuildTools"):
        roots.append(pathlib.Path(r"C:\Program Files\Microsoft Visual Studio\2022") / edition)

    vcvars = next(
        (p for r in roots if (p := r / "VC" / "Auxiliary" / "Build" / "vcvars64.bat").exists()),
        None,
    )
    if vcvars is None:
        return  # let nvcc produce its own error

    # Drop Git's unix bin dirs (their `sort`/`find` break vcvars) and dedupe, so
    # the PATH vcvars hands back stays under cmd.exe's 8191-char command limit.
    seen: set[str] = set()
    clean: list[str] = []
    for p in os.environ.get("PATH", "").split(os.pathsep):
        low = p.lower()
        if not p or low in seen or r"git\usr" in low or r"git\mingw64" in low:
            continue
        seen.add(low)
        clean.append(p)
    env = dict(os.environ, PATH=os.pathsep.join(clean))

    # The command must be one string: subprocess's Windows list-quoting would
    # escape the inner quotes around the .bat path and cmd would not find it.
    cmdline = f'cmd.exe /s /c ""{vcvars}" >nul 2>&1 && set"'
    res = subprocess.run(cmdline, capture_output=True, text=True, env=env, timeout=180)
    if res.returncode != 0:
        return
    for line in res.stdout.splitlines():
        key, sep, val = line.partition("=")
        if sep and key.upper() not in ("PROMPT", "PYTHONPATH"):
            os.environ[key] = val


def _arch_flags() -> list[str]:
    major, minor = torch.cuda.get_device_capability(0)
    return [f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"]


def load_extension(verbose: bool = False, force: bool = False):
    """Compile (or fetch from cache) the CUDA extension. Returns None on failure."""
    global _EXT, _LOAD_ERROR
    if _EXT is not None and not force:
        return _EXT
    if _LOAD_ERROR is not None and not force:
        return None

    from torch.utils.cpp_extension import load

    _ensure_msvc_env()

    _BUILD.mkdir(parents=True, exist_ok=True)
    nvcc_flags = [
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "-lineinfo",  # Nsight Compute source-level attribution
        *_arch_flags(),
    ]
    if os.name == "nt":
        nvcc_flags.append("-allow-unsupported-compiler")
        cxx_flags = ["/O2", "/std:c++17"]
    else:
        cxx_flags = ["-O3", "-std=c++17"]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _EXT = load(
                name="pagedattn_cuda",
                sources=[str(_SRC)],
                extra_cuda_cflags=nvcc_flags,
                extra_cflags=cxx_flags,
                build_directory=str(_BUILD),
                verbose=verbose,
            )
        _LOAD_ERROR = None
        return _EXT
    except Exception as exc:  # pragma: no cover - environment dependent
        _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        return None


def is_available() -> bool:
    return load_extension() is not None


def load_error() -> str | None:
    return _LOAD_ERROR


# Kernel variants compiled into the extension. `variant=` selects one; the
# names are what the sweep and the README use.
VARIANTS = {
    "v1_naive": 0,  # 4 warps, no unroll: 2 memory ops in flight per thread
    "v2_unroll": 1,  # 4 warps, unroll 4: 8 in flight
    "v3_tuned": 2,  # 8 warps, unroll 4: 8 in flight, double the warps
}
# Measured on RTX 3050 (sm_86): the unroll is worth 5-8 points of peak bandwidth,
# but doubling the warps on top of it is consistently a slight *loss* -- see the
# README's "Occupancy is not the goal". So v2, not v3, is the default.
DEFAULT_VARIANT = "v2_unroll"

# (head_dim, gqa_group) pairs compiled into the extension. The Triton kernel
# takes both as constexpr and specializes on demand; the CUDA kernel has to be
# instantiated ahead of time, so this list is the honest boundary of what it can
# serve. Callers should consult `supports_shape` rather than discover it from a
# TORCH_CHECK deep in a launch.
SUPPORTED_SHAPES = frozenset({(128, 4), (128, 8), (64, 4), (64, 7), (64, 8)})


def supports_shape(head_dim: int, group: int) -> tuple[bool, str]:
    if (head_dim, group) in SUPPORTED_SHAPES:
        return True, ""
    return False, (
        f"CUDA kernel has no instantiation for (head_dim={head_dim}, "
        f"group={group}); compiled: {sorted(SUPPORTED_SHAPES)}. "
        "The Triton kernel handles this shape.")


def paged_decode_cuda(
    q: torch.Tensor,
    cache: PagedKVCache,
    out: torch.Tensor | None = None,
    num_splits: int | None = None,
    variant: str | int = DEFAULT_VARIANT,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Paged GQA decode attention, hand-written CUDA. q: [B, Hq, D] fp16."""
    ext = load_extension()
    if ext is None:
        raise RuntimeError(f"CUDA extension unavailable: {_LOAD_ERROR}")

    b, hq, d = q.shape
    shape = cache.shape
    ok, why = supports_shape(d, shape.gqa_group)
    if not ok:
        raise ValueError(why)
    if out is None:
        out = torch.empty((b, hq, d), dtype=q.dtype, device=q.device)

    if num_splits is None:
        # See the note in triton_decode.paged_decode_triton: this sync is why the
        # split count is a caller-supplied argument on the hot path.
        max_seq = int(cache.seq_lens.max().item())
        sm = torch.cuda.get_device_properties(q.device).multi_processor_count
        # The CUDA kernel consumes 8 tokens per iteration, so its "tile" for the
        # starvation heuristic is 8 rather than the Triton BLOCK_N.
        num_splits = pick_num_splits(b, shape.num_kv_heads, max_seq, sm, block_n=256)

    k_scale = cache.k_scale if cache.k_scale is not None else _EMPTY
    v_scale = cache.v_scale if cache.v_scale is not None else _EMPTY

    if num_splits <= 1:
        part_out = part_lse = _EMPTY
    else:
        part_out, part_lse = _WS.get(b, hq, num_splits, d, q.device)

    vid = VARIANTS[variant] if isinstance(variant, str) else int(variant)
    ext.paged_decode(
        q, cache.k_cache, cache.v_cache, k_scale, v_scale,
        cache.block_table, cache.seq_lens, out, part_out, part_lse,
        shape.softmax_scale if sm_scale is None else float(sm_scale),
        _KV_CODE[cache.kv_dtype], int(num_splits), vid,
    )
    return out

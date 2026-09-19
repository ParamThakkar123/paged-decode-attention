"""GPU timing that does not lie to you.

Two things ruin decode-kernel microbenchmarks and both are handled here:

1. *L2 residency.* At batch=1, seqlen=1k the whole KV working set is 4 MB. Run
   that in a tight loop on a card with a 2 MB L2 and you will measure a number
   that no real server ever sees. `flush_l2=True` scrubs the cache between
   iterations, and `l2_resident_fraction` reports how much of the working set
   would have fit, so suspicious rows can be flagged rather than silently
   believed.

2. *CUDA event overhead.* A single event pair costs a few microseconds, which is
   10-20% of a short decode kernel. We time a run of `inner` launches between one
   event pair and divide, so the per-call number still includes launch overhead
   (a serving runtime pays that too) without the measurement tax.
"""

from __future__ import annotations

import statistics
from typing import Callable

import torch


def l2_bytes(device: int = 0) -> int:
    return int(torch.cuda.get_device_properties(device).L2_cache_size)


class _L2Flusher:
    """A buffer comfortably larger than L2, rewritten between timed runs."""

    def __init__(self, device: int = 0) -> None:
        n = max(4 * l2_bytes(device), 16 << 20)
        self.buf = torch.empty(n // 4, dtype=torch.int32, device="cuda")

    def flush(self) -> None:
        self.buf.zero_()


_FLUSHER: _L2Flusher | None = None


def _flusher() -> _L2Flusher:
    global _FLUSHER
    if _FLUSHER is None:
        _FLUSHER = _L2Flusher()
    return _FLUSHER


def bench(
    fn: Callable[[], object],
    warmup: int = 20,
    reps: int = 30,
    inner: int | None = None,
    target_ms: float = 2.0,
    flush_l2: bool = True,
    budget_ms: float = 600.0,
) -> dict[str, float]:
    """Return {'ms': median, 'ms_p20', 'ms_p80', 'inner', 'reps'} per single call.

    `budget_ms` bounds the wall time spent on one measurement. Without it the
    slow dense baselines dominate a sweep: PyTorch's math SDPA takes ~350 ms per
    call at batch 64 / context 2k, and 20 warmups plus 30 reps of that is 20
    seconds for a single cell of the table. Fast kernels still get the full rep
    count -- the clamp only bites when a call is already slow enough that its
    run-to-run variance is negligible.
    """
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    # Always calibrate: the per-call cost decides both `inner` and how many
    # repetitions we can afford.
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(3):
        fn()
    end.record()
    torch.cuda.synchronize()
    one_ms = max(start.elapsed_time(end) / 3, 1e-4)

    if inner is None:
        inner = max(1, min(200, int(target_ms / one_ms)))

    per_run_ms = one_ms * inner
    reps = max(5, min(reps, int(budget_ms / max(per_run_ms, 1e-3))))
    warmup = max(2, min(warmup, int(budget_ms / (4 * max(one_ms, 1e-3)))))

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(reps):
        if flush_l2:
            _flusher().flush()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / inner)

    samples.sort()
    return {
        "ms": statistics.median(samples),
        "ms_p20": samples[int(0.2 * (len(samples) - 1))],
        "ms_p80": samples[int(0.8 * (len(samples) - 1))],
        "inner": float(inner),
        "reps": float(reps),
    }


# ---------------------------------------------------------------------------
# peak bandwidth
# ---------------------------------------------------------------------------


def theoretical_peak_gbs(device: int = 0) -> float | None:
    """Theoretical DRAM bandwidth from the memory clock and bus width, if torch
    exposes them (it does not on every build)."""
    props = torch.cuda.get_device_properties(device)
    clock_khz = getattr(props, "memory_clock_rate", None)  # kHz
    bus_bits = getattr(props, "memory_bus_width", None)
    if not clock_khz or not bus_bits:
        return None
    # GDDR6 transfers on both edges.
    return clock_khz * 1e3 * 2 * (bus_bits / 8) / 1e9


def triton_usable() -> bool:
    """Can Triton actually compile and launch, not merely be imported?

    `import triton` succeeds on a machine with no host C compiler; the failure
    only appears at first launch, when Triton builds its runtime shim:
    "RuntimeError: Failed to find C compiler." Checking the import alone
    therefore reports a working Triton on an environment where every kernel
    will fail -- which is exactly what the WSL2 image used for the vLLM
    baselines does.
    """
    global _TRITON_OK
    if _TRITON_OK is not None:
        return _TRITON_OK
    import os

    if _probe_kernel is None:
        _TRITON_OK = False
        return False
    try:
        x = torch.zeros(8, dtype=torch.int32, device="cuda")
        _probe_kernel[(8,)](x)
        torch.cuda.synchronize()
        _TRITON_OK = bool(int(x.sum()) == 8)
    except Exception as exc:
        if os.environ.get("PAGEDATTN_VERBOSE"):
            print(f"[triton_usable] unavailable: {type(exc).__name__}: {exc}")
        _TRITON_OK = False
    return _TRITON_OK


_TRITON_OK: bool | None = None

# The probe kernel must live at module scope. Triton resolves a kernel's free
# names from its *module* globals, so a kernel defined inside a function sees
# neither a locally-imported `tl` nor anything else local, and fails with
# `NameError('tl is not defined')` — which reads exactly like "Triton is
# broken" and silently drops our kernel from any comparison that checks this.
try:
    import triton as _triton
    import triton.language as tl

    @_triton.jit
    def _probe_kernel(X):
        tl.store(X + tl.program_id(0), 1)

except Exception:  # triton not importable at all
    _probe_kernel = None


def measured_peak_gbs_torch(bytes_total: int = 512 << 20, device: int = 0) -> float:
    """Streaming-read peak using only torch ops -- no Triton, no C compiler.

    Needed because Triton JITs through a host C compiler, and the WSL2 image
    used for the vLLM baselines has none. A large fp16 reduction is a good
    enough proxy: it is DRAM-bound and ATen's reduction is well optimized, so it
    lands within a couple of percent of the tuned Triton probe.
    """
    free, _ = torch.cuda.mem_get_info(device)
    nbytes = min(bytes_total, int(free * 0.35))
    n = (nbytes // 2) & ~65535
    x = torch.ones(n, dtype=torch.float16, device="cuda")

    best = 0.0
    for _ in range(3):
        res = bench(lambda: x.sum(dtype=torch.float32), warmup=5, reps=10, flush_l2=False)
        best = max(best, (n * 2) / (res["ms"] * 1e-3) / 1e9)
    del x
    torch.cuda.empty_cache()
    return best


def measured_peak_gbs(
    bytes_total: int = 512 << 20, device: int = 0, verbose: bool = False
) -> float:
    """Best streaming-read bandwidth this GPU will give us.

    A pure read is the right probe: decode attention reads the KV cache and
    writes essentially nothing, so a read+write copy benchmark would understate
    the ceiling we should be measured against.

    We sweep tile size, vector width and warp count and keep the best, because a
    single arbitrary configuration can fall 15-20% short of what the memory
    system can actually deliver -- and a probe that undershoots produces
    "% of peak" numbers above 100, which is worse than useless.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def _read_kernel(X, Out, n, BLOCK: tl.constexpr, UNROLL: tl.constexpr):
        pid = tl.program_id(0)
        base = pid * BLOCK * UNROLL
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for u in tl.static_range(UNROLL):
            offs = base + u * BLOCK + tl.arange(0, BLOCK)
            acc += tl.load(X + offs, mask=offs < n, other=0.0).to(tl.float32)
        # Guarded so the loads cannot be eliminated, but never actually taken.
        if tl.sum(acc) == 1.2345e30:
            tl.store(Out + pid, 1.0)

    free, _ = torch.cuda.mem_get_info(device)
    nbytes = min(bytes_total, int(free * 0.35))
    n = (nbytes // 2) & ~65535
    x = torch.ones(n, dtype=torch.float16, device="cuda")
    out = torch.zeros(4096, dtype=torch.float32, device="cuda")

    if not triton_usable():
        raise RuntimeError(
            "Triton cannot compile here; use measured_peak_gbs_torch() instead")

    best = 0.0
    for block in (1024, 2048, 4096):
        for unroll in (1, 2, 4):
            for warps in (4, 8):
                grid = (triton.cdiv(n, block * unroll),)

                def run(b=block, u=unroll, w=warps, g=grid):
                    _read_kernel[g](x, out, n, BLOCK=b, UNROLL=u, num_warps=w)

                try:
                    res = bench(run, warmup=5, reps=10, flush_l2=False)
                except Exception:
                    continue
                gbs = (n * 2) / (res["ms"] * 1e-3) / 1e9
                if verbose:
                    print(f"    block={block:5d} unroll={unroll} warps={warps}: {gbs:6.1f} GB/s")
                best = max(best, gbs)

    del x, out
    torch.cuda.empty_cache()
    return best


def _sane_peak(measured: float, label: str) -> float:
    """Reject a bandwidth probe that is obviously wrong.

    The probe occasionally reads far too low -- another process still holding
    the GPU, a power-state transition, a scheduling hiccup. Every "% of peak" in
    the repo divides by this number, so a bad probe does not produce a slightly
    off result, it produces an impossible one: a 380 us kernel once reported
    396 % of peak. Anything below 60 % of the theoretical ceiling is not a slow
    GPU, it is a broken measurement, and callers are told rather than left to
    publish it.
    """
    theo = theoretical_peak_gbs()
    if theo and measured < 0.60 * theo:
        raise RuntimeError(
            f"{label} measured {measured:.1f} GB/s, under 60% of the {theo:.1f} GB/s "
            "theoretical ceiling -- the GPU was almost certainly busy. Re-run with "
            "nothing else using it.")
    return measured


def measured_peak_gbs_checked(**kw) -> float:
    """`measured_peak_gbs` with retries and a sanity check."""
    last = 0.0
    for attempt in range(3):
        try:
            last = measured_peak_gbs(**kw)
            return _sane_peak(last, "streaming-read probe")
        except RuntimeError:
            if attempt == 2:
                raise
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    return last

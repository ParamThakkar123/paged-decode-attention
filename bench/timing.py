"""GPU timing for decode microbenchmarks.

Two things that would otherwise ruin the numbers:

1. *L2 residency.* At batch 1 / seqlen 1k the KV working set is 4 MB, which a
   tight loop keeps in a 2 MB L2 -- a number no server ever sees. `flush_l2`
   scrubs the cache between iterations.

2. *Event overhead.* An event pair costs a few microseconds, 10-20% of a short
   decode kernel. Timing `inner` launches per pair amortizes it while keeping
   launch cost in the per-call number, which a serving runtime also pays.
"""

from __future__ import annotations

import statistics
from typing import Callable

import torch


def l2_bytes(device: int = 0) -> int:
    return int(torch.cuda.get_device_properties(device).L2_cache_size)


class _L2Flusher:
    """A buffer larger than L2, rewritten between timed runs."""

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
    """Return {'ms': median, 'ms_p20', 'ms_p80', 'inner', 'reps'} per call.

    `budget_ms` bounds wall time per measurement. Without it the slow dense
    baselines dominate: math SDPA takes ~350 ms per call at batch 64 / ctx 2k,
    so 20 warmups + 30 reps is 20 s for one cell. The clamp only bites once a
    call is slow enough that run-to-run variance is negligible.
    """
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    # Calibrate: per-call cost decides both `inner` and the rep budget.
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
    """Theoretical DRAM bandwidth from clock and bus width, if torch exposes
    them (not on every build)."""
    props = torch.cuda.get_device_properties(device)
    clock_khz = getattr(props, "memory_clock_rate", None)  # kHz
    bus_bits = getattr(props, "memory_bus_width", None)
    if not clock_khz or not bus_bits:
        return None
    # GDDR6 transfers on both edges.
    return clock_khz * 1e3 * 2 * (bus_bits / 8) / 1e9


def triton_usable() -> bool:
    """Can Triton compile and launch, not merely be imported?

    `import triton` succeeds without a host C compiler; the failure appears at
    first launch ("Failed to find C compiler"). Checking the import alone
    reports a working Triton on the WSL2 image, where every kernel fails.
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

# Must be module scope: Triton resolves free names from module globals, so a
# kernel defined inside a function fails with `NameError('tl is not defined')`,
# which looks exactly like "Triton is broken".
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

    A large fp16 reduction is DRAM-bound and well optimized, landing within a
    couple of percent of the tuned Triton probe below.
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
    """Best streaming-read bandwidth this GPU will give.

    A pure read is the right probe: decode attention reads KV and writes
    almost nothing, so a copy benchmark would understate the ceiling.

    Sweeps tile size, vector width and warp count because one arbitrary config
    can fall 15-20% short, and a probe that undershoots yields "% of peak"
    above 100.
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
        # Guarded so the loads survive DCE, but never taken.
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

    The probe occasionally reads far too low (another process on the GPU, a
    power-state transition). Every "% of peak" divides by it, so a bad probe
    gives impossible results -- a 380 us kernel once reported 396 % of peak.
    Under 60 % of theoretical is a broken measurement, not a slow GPU.
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

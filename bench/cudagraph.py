"""CUDA-graph capture for the decode kernels.

Why this exists. On this machine (Windows, WDDM driver model) a single kernel
launch costs 40-230 microseconds of *host* time before the GPU does anything.
A 4 MB decode step should take ~25 us. So below roughly batch 8 the eager
measurement is a benchmark of the Windows launch path, not of the kernel, and
no amount of kernel tuning moves it.

Real serving runtimes hit the same wall -- decode is thousands of tiny launches
per second -- and they solve it the same way: capture the whole decode step into
a CUDA graph once and replay it. vLLM does this by default for decode batches.
Capturing here is therefore not a benchmarking trick, it is the configuration a
serving runtime actually runs, and it is the only way to see the kernel's real
cost on this platform.

The sweep reports both numbers, `eager` and `graphed`, because the gap between
them is itself the result: it says how much of a decode step is launch overhead.
"""

from __future__ import annotations

from typing import Callable

import torch


class GraphedOp:
    """Capture a zero-argument op into a CUDA graph and replay it.

    All tensors the op touches must be allocated before capture and must not be
    reallocated afterwards -- the graph records the pointers, not the tensors.
    Callers therefore pass a closure over already-allocated buffers and a
    preallocated `out`.
    """

    def __init__(self, fn: Callable[[], object], warmup: int = 5) -> None:
        self.fn = fn
        # Warm up on a side stream: this compiles the Triton kernel, populates
        # any lazily-allocated workspace, and settles cuBLAS handles, none of
        # which may happen during capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            fn()
        torch.cuda.synchronize()

    def __call__(self) -> None:
        self.graph.replay()


def try_graph(fn: Callable[[], object], warmup: int = 5) -> tuple[GraphedOp | None, str]:
    """Capture `fn`, or explain why it could not be captured."""
    try:
        return GraphedOp(fn, warmup=warmup), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"[:200]

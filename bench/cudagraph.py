"""CUDA-graph capture for the decode kernels.

On Windows (WDDM) a kernel launch costs 40-230 us of *host* time, against a
~25 us decode step. Below about batch 8, eager timing measures the launch path
rather than the kernel.

Serving runtimes hit the same wall and solve it the same way -- vLLM captures
decode batches into graphs by default -- so capturing here is the configuration
a runtime actually runs, not a benchmarking trick.

The sweep reports `eager` and `graphed` both, because the gap is itself the
result: how much of a decode step is launch overhead (README section 5.1).
"""

from __future__ import annotations

from typing import Callable

import torch


class GraphedOp:
    """Capture a zero-argument op into a CUDA graph and replay it.

    The graph records pointers, not tensors, so every tensor the op touches must
    be allocated before capture and never reallocated. Callers pass a closure
    over existing buffers and a preallocated `out`.
    """

    def __init__(self, fn: Callable[[], object], warmup: int = 5) -> None:
        self.fn = fn
        # Side-stream warmup: compile the kernel, populate lazy workspaces and
        # settle cuBLAS handles, none of which may happen during capture.
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

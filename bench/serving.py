"""Continuous-batching decode loop: block allocator, scheduler, graph runner.

A microbenchmark on a fixed (batch, context) rectangle answers the wrong
question. Real decode batches are ragged, grow a token per step, retire and
refill at different times, and reuse the blocks the retired ones left behind --
all four change what the kernel sees.

  BlockAllocator  - free list over physical KV pages. LIFO, which is what makes
                    a long-running server's block tables fragmented.
  Scheduler       - admits sequences up to a batch and KV budget, grows the
                    running ones each step, retires finished ones.
  DecodeRunner    - static buffers plus one CUDA graph per batch bucket.

The graph part is the subtle one: a graph bakes in shapes, pointers and grid
dimensions but not tensor *contents*, so the block table and sequence lengths
can change every step as long as they are written in place. Only batch size has
to be bucketed. That is vLLM's trick, and it is why `seq_lens` lives on the GPU
and is never read back.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Callable

import torch

from pagedattn.cache import PagedKVCache
from pagedattn.config import ModelShape, torch_dtype


class OutOfBlocks(RuntimeError):
    pass


class BlockAllocator:
    """Free list over physical KV pages."""

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        # Reversed so the first pops are low indices; a few alloc/free rounds
        # then scramble it, like a warm server.
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def utilization(self) -> float:
        return 1.0 - len(self._free) / self.num_blocks

    def alloc(self, n: int = 1) -> list[int]:
        if n > len(self._free):
            raise OutOfBlocks(f"need {n} blocks, {len(self._free)} free")
        return [self._free.pop() for _ in range(n)]

    def free(self, blocks: list[int]) -> None:
        self._free.extend(blocks)


@dataclasses.dataclass
class Sequence:
    sid: int
    prompt_len: int
    target_len: int  # prompt + tokens to generate
    seq_len: int
    blocks: list[int] = dataclasses.field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.seq_len >= self.target_len


class Scheduler:
    """Admission, growth and retirement for a continuous-batching decode loop."""

    def __init__(
        self,
        allocator: BlockAllocator,
        block_size: int,
        max_batch: int,
        prompt_lens: Callable[[], int],
        gen_lens: Callable[[], int],
    ) -> None:
        self.alloc = allocator
        self.block_size = block_size
        self.max_batch = max_batch
        self.prompt_lens = prompt_lens
        self.gen_lens = gen_lens
        self.running: list[Sequence] = []
        self._next_sid = 0
        self.completed = 0
        self.rejected = 0

    def _blocks_for(self, n_tokens: int) -> int:
        return (n_tokens + self.block_size - 1) // self.block_size

    def admit(self) -> None:
        """Fill the batch up to max_batch, as far as free blocks allow."""
        while len(self.running) < self.max_batch:
            plen = self.prompt_lens()
            glen = self.gen_lens()
            # Reserve the whole generation up front. A real scheduler would
            # allocate lazily and preempt; that is a different project.
            need = self._blocks_for(plen + glen)
            if need > self.alloc.num_free:
                self.rejected += 1
                return
            seq = Sequence(self._next_sid, plen, plen + glen, plen,
                           self.alloc.alloc(need))
            self._next_sid += 1
            self.running.append(seq)

    def step_grow(self) -> None:
        """One decoded token per running sequence; retire the finished ones."""
        still: list[Sequence] = []
        for s in self.running:
            s.seq_len += 1
            if s.done:
                self.alloc.free(s.blocks)
                self.completed += 1
            else:
                still.append(s)
        self.running = still


class DecodeRunner:
    """Static KV buffers plus one CUDA graph per batch bucket."""

    def __init__(
        self,
        shape: ModelShape,
        num_blocks: int,
        block_size: int,
        max_batch: int,
        max_blocks_per_seq: int,
        kv_dtype: str = "fp16",
        device: str = "cuda",
    ) -> None:
        self.shape = shape
        self.block_size = block_size
        self.max_batch = max_batch
        h, d = shape.num_kv_heads, shape.head_dim
        dt = torch_dtype(kv_dtype)  # type: ignore[arg-type]

        self.k_cache = torch.zeros((num_blocks, block_size, h, d), dtype=dt, device=device)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.block_table = torch.zeros((max_batch, max_blocks_per_seq),
                                       dtype=torch.int32, device=device)
        # Padded slots get seq_len 1, not 0, to keep every softmax denominator
        # non-zero for trivial cost.
        self.seq_lens = torch.ones((max_batch,), dtype=torch.int32, device=device)
        self.q = torch.randn((max_batch, shape.num_q_heads, d),
                             dtype=torch.float16, device=device)
        self.out = torch.empty_like(self.q)
        self.kv_dtype = kv_dtype
        self._graphs: dict[tuple[int, int], object] = {}

    def buckets(self, granularity: int = 8) -> list[int]:
        """Batch sizes we capture graphs for. A step pads up to the next one."""
        out, b = [], granularity
        while b < self.max_batch:
            out.append(b)
            b *= 2
        out.append(self.max_batch)
        return sorted({min(x, self.max_batch) for x in ([1, 2, 4] + out)})

    def _view(self, batch: int) -> PagedKVCache:
        return PagedKVCache(
            shape=self.shape, block_size=self.block_size, kv_dtype=self.kv_dtype,  # type: ignore[arg-type]
            k_cache=self.k_cache, v_cache=self.v_cache, k_scale=None, v_scale=None,
            block_table=self.block_table[:batch], seq_lens=self.seq_lens[:batch],
        )

    def make_fn(self, batch: int, kernel: Callable, num_splits: int) -> Callable[[], object]:
        cache = self._view(batch)
        q = self.q[:batch]
        out = self.out[:batch]
        return lambda: kernel(q, cache, out=out, num_splits=num_splits)

    def graph_for(self, batch: int, kernel: Callable, num_splits: int):
        from bench.cudagraph import try_graph

        key = (batch, num_splits)
        if key not in self._graphs:
            g, err = try_graph(self.make_fn(batch, kernel, num_splits))
            self._graphs[key] = g if g is not None else err
        return self._graphs[key]

    def load_batch(self, seqs: list[Sequence]) -> None:
        """Write this step's block table and lengths into the static buffers."""
        n = len(seqs)
        max_blk = self.block_table.shape[1]
        bt = torch.zeros((n, max_blk), dtype=torch.int32)
        lens = torch.empty((n,), dtype=torch.int32)
        for i, s in enumerate(seqs):
            nb = min(len(s.blocks), max_blk)
            bt[i, :nb] = torch.tensor(s.blocks[:nb], dtype=torch.int32)
            lens[i] = min(s.seq_len, nb * self.block_size)
        self.block_table[:n].copy_(bt, non_blocking=True)
        self.seq_lens[:n].copy_(lens, non_blocking=True)
        # Padding: one token at block 0 -- cheap and safe.
        if n < self.max_batch:
            self.seq_lens[n:].fill_(1)


def poisson_lengths(rng: random.Random, lo: int, hi: int) -> Callable[[], int]:
    """Uniform-in-log length sampler -- real traffic is heavy-tailed, not uniform."""
    import math

    llo, lhi = math.log(lo), math.log(hi)

    def sample() -> int:
        return int(math.exp(rng.uniform(llo, lhi)))

    return sample

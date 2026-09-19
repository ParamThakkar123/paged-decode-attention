"""Paged-KV GQA decode attention in Triton.

One kernel, three constexpr switches, so every optimization claim in the README
is an A/B against the *same* code path rather than against a different kernel:

  PER_PAGE_BT : load the block table once per page instead of once per token.
  SPLIT_KV    : partition the KV range across CTAs (FlashDecoding), then reduce.
  KV_DTYPE    : 0 = fp16, 1 = fp8_e5m2, 2 = int8 with per-(token,head) scales.

Parallelization: one program per (batch, kv_head[, split]). All GQA_GROUP query
heads that share a KV head are handled by the same program, so each K/V byte is
read from DRAM once and reused `group` times out of registers. That reuse is the
whole point of GQA for decode, and it is why the achieved-bandwidth numbers in
the README are computed against KV bytes rather than against Q*K flops.

The query tile is padded from GQA_GROUP (4 for Llama-3-8B) up to 16 because
`tl.dot` requires M >= 16. That wastes 4x of the tensor-core issue slots, and
Nsight confirms it costs nothing: at batch 32 / context 4k this kernel runs at
97.6% of DRAM peak with the tensor pipe only 14.5% busy and overall SM
throughput at 17.5%. There is nothing to reclaim by unpadding -- the memory
system is the wall.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .cache import PagedKVCache

LOG2E = tl.constexpr(1.4426950408889634)

# KV_DTYPE codes, kept in sync with config.KVDType.
_KV_CODE = {"fp16": 0, "bf16": 0, "fp8_e5m2": 1, "int8": 2}


@triton.jit
def _paged_decode_kernel(
    Q,  # [B, Hq, D]
    K_cache,  # [NB, PAGE, Hkv, D]
    V_cache,
    K_scale,  # [NB, PAGE, Hkv] fp16 (int8 only; else aliased to K_cache)
    V_scale,
    Out,  # [B, Hq, D]           when SPLIT_KV == 0
    PartOut,  # [B, Hq, SPLITS, D] fp32  when SPLIT_KV == 1
    PartLse,  # [B, Hq, SPLITS]    fp32  when SPLIT_KV == 1
    BlockTables,  # [B, MAXBLK] int32
    SeqLens,  # [B] int32
    sm_scale,
    stride_qb,
    stride_qh,
    stride_kn,
    stride_kt,
    stride_kh,
    stride_sn,
    stride_st,
    stride_ob,
    stride_oh,
    stride_pb,
    stride_ph,
    stride_ps,
    stride_lb,
    stride_lh,
    stride_btb,
    num_splits,
    HEAD_DIM: tl.constexpr,
    PAGE: tl.constexpr,
    GROUP: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_DTYPE: tl.constexpr,
    PER_PAGE_BT: tl.constexpr,
    SPLIT_KV: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)  # kv head
    pid_s = tl.program_id(2) if SPLIT_KV else 0

    seq_len = tl.load(SeqLens + pid_b).to(tl.int32)

    # ---- KV range owned by this program ------------------------------------
    if SPLIT_KV:
        # Chunk boundaries are BLOCK_N-aligned so every program's inner loop
        # keeps the same tile shape and the tail mask stays cheap.
        chunk = tl.cdiv(tl.cdiv(seq_len, BLOCK_N), num_splits) * BLOCK_N
        lo = pid_s * chunk
        hi = tl.minimum(lo + chunk, seq_len)
    else:
        lo = 0
        hi = seq_len

    offs_d = tl.arange(0, HEAD_DIM)
    offs_g = tl.arange(0, GROUP_PAD)
    g_mask = offs_g < GROUP
    q_heads = pid_h * GROUP + offs_g

    q = tl.load(
        Q + pid_b * stride_qb + q_heads[:, None] * stride_qh + offs_d[None, :],
        mask=g_mask[:, None],
        other=0.0,
    )

    m_i = tl.full([GROUP_PAD], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([GROUP_PAD], dtype=tl.float32)
    acc = tl.zeros([GROUP_PAD, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * LOG2E

    # An empty range happens when seq_len is short relative to num_splits.
    if lo < hi:
        for start_n in tl.range(lo, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < hi

            # ---- paged address computation ---------------------------------
            if PER_PAGE_BT:
                # BLOCK_N spans BLOCK_N/PAGE pages. Load one block-table entry
                # per page and broadcast it across the page's tokens, instead of
                # issuing BLOCK_N redundant loads that only L1 saves us from.
                offs_p = tl.arange(0, BLOCK_N // PAGE)
                page_id = (start_n // PAGE) + offs_p
                p_mask = (page_id * PAGE) < hi
                phys_p = tl.load(
                    BlockTables + pid_b * stride_btb + page_id, mask=p_mask, other=0
                ).to(tl.int32)
                phys = tl.reshape(
                    tl.broadcast_to(phys_p[:, None], (BLOCK_N // PAGE, PAGE)), (BLOCK_N,)
                )
                tok = tl.reshape(
                    tl.broadcast_to(tl.arange(0, PAGE)[None, :], (BLOCK_N // PAGE, PAGE)),
                    (BLOCK_N,),
                )
            else:
                logical_blk = offs_n // PAGE
                tok = offs_n % PAGE
                phys = tl.load(
                    BlockTables + pid_b * stride_btb + logical_blk, mask=n_mask, other=0
                ).to(tl.int32)

            kv_off = (
                phys[:, None] * stride_kn
                + tok[:, None] * stride_kt
                + pid_h * stride_kh
                + offs_d[None, :]
            )

            k = tl.load(K_cache + kv_off, mask=n_mask[:, None], other=0.0)
            v = tl.load(V_cache + kv_off, mask=n_mask[:, None], other=0.0)

            if KV_DTYPE == 2:
                s_off = phys * stride_sn + tok * stride_st + pid_h
                ks = tl.load(K_scale + s_off, mask=n_mask, other=0.0).to(tl.float32)
                vs = tl.load(V_scale + s_off, mask=n_mask, other=0.0).to(tl.float32)
                k = (k.to(tl.float32) * ks[:, None]).to(tl.float16)
                v = (v.to(tl.float32) * vs[:, None]).to(tl.float16)
            else:
                # fp8_e5m2 -> fp16 is a shift on Ampere (shared 5-bit exponent).
                k = k.to(tl.float16)
                v = v.to(tl.float16)

            # ---- online softmax --------------------------------------------
            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])

            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.float16), v, acc, out_dtype=tl.float32)
            m_i = m_new

    # ---- epilogue -----------------------------------------------------------
    if SPLIT_KV:
        empty = lo >= hi
        l_safe = tl.where(l_i > 0.0, l_i, 1.0)
        acc = acc / l_safe[:, None]
        lse = tl.where(empty | (l_i <= 0.0), float("-inf"), m_i + tl.log2(l_safe))

        tl.store(
            PartOut
            + pid_b * stride_pb
            + q_heads[:, None] * stride_ph
            + pid_s * stride_ps
            + offs_d[None, :],
            acc,
            mask=g_mask[:, None],
        )
        tl.store(
            PartLse + pid_b * stride_lb + q_heads * stride_lh + pid_s,
            lse,
            mask=g_mask,
        )
    else:
        acc = acc / l_i[:, None]
        tl.store(
            Out + pid_b * stride_ob + q_heads[:, None] * stride_oh + offs_d[None, :],
            acc.to(Out.dtype.element_ty),
            mask=g_mask[:, None],
        )


@triton.jit
def _split_reduce_kernel(
    PartOut,  # [B, Hq, SPLITS, D] fp32
    PartLse,  # [B, Hq, SPLITS]    fp32
    Out,  # [B, Hq, D]
    stride_pb,
    stride_ph,
    stride_ps,
    stride_lb,
    stride_lh,
    stride_ob,
    stride_oh,
    num_splits,
    HEAD_DIM: tl.constexpr,
    SPLIT_PAD: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_s = tl.arange(0, SPLIT_PAD)
    s_mask = offs_s < num_splits
    offs_d = tl.arange(0, HEAD_DIM)

    lse = tl.load(
        PartLse + pid_b * stride_lb + pid_h * stride_lh + offs_s,
        mask=s_mask,
        other=float("-inf"),
    )
    m = tl.max(lse, 0)
    w = tl.where(s_mask, tl.exp2(lse - m), 0.0)  # exp2: lse is kept in log2 space
    denom = tl.sum(w, 0)

    part = tl.load(
        PartOut
        + pid_b * stride_pb
        + pid_h * stride_ph
        + offs_s[:, None] * stride_ps
        + offs_d[None, :],
        mask=s_mask[:, None],
        other=0.0,
    )
    out = tl.sum(part * w[:, None], 0) / denom
    tl.store(Out + pid_b * stride_ob + pid_h * stride_oh + offs_d, out.to(Out.dtype.element_ty))


# ----------------------------------------------------------------------------
# python wrappers
# ----------------------------------------------------------------------------


def pick_num_splits(
    batch: int, num_kv_heads: int, max_seq_len: int, sm_count: int, block_n: int
) -> int:
    """How many KV splits before the GPU stops being starved.

    At batch=1 the un-split kernel launches only num_kv_heads=8 CTAs, which on a
    16-SM card leaves half the GPU idle no matter how long the context is. We
    split until we have ~2 CTAs per SM, but never so far that a split covers
    fewer than 2 BLOCK_N tiles (the fixed per-CTA cost stops paying for itself).
    """
    base_ctas = batch * num_kv_heads
    if base_ctas >= 2 * sm_count:
        return 1
    want = -(-2 * sm_count // base_ctas)
    max_useful = max(1, (max_seq_len // block_n) // 2)
    n = max(1, min(want, max_useful, 64))
    return 1 << (n.bit_length() - 1)  # round down to a power of two


class _Workspace:
    """Reused split-KV scratch, so the sweep does not re-allocate every call."""

    def __init__(self) -> None:
        self.part_out: torch.Tensor | None = None
        self.part_lse: torch.Tensor | None = None

    def get(self, b: int, hq: int, splits: int, d: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        need = (b, hq, splits, d)
        if self.part_out is None or tuple(self.part_out.shape) != need:
            self.part_out = torch.empty(need, dtype=torch.float32, device=device)
            self.part_lse = torch.empty((b, hq, splits), dtype=torch.float32, device=device)
        assert self.part_lse is not None
        return self.part_out, self.part_lse


_WS = _Workspace()


def paged_decode_triton(
    q: torch.Tensor,
    cache: PagedKVCache,
    out: torch.Tensor | None = None,
    block_n: int = 64,
    num_splits: int | None = None,
    per_page_bt: bool = True,
    num_warps: int = 4,
    num_stages: int = 3,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Paged GQA decode attention. q: [B, Hq, D] fp16 -> [B, Hq, D] fp16.

    `sm_scale` overrides the default 1/sqrt(head_dim). A serving runtime supplies
    its own scale (it may fold in other factors), so it must not be re-derived
    here from the shape.
    """
    b, hq, d = q.shape
    shape = cache.shape
    hkv = shape.num_kv_heads
    group = shape.gqa_group
    group_pad = max(16, triton.next_power_of_2(group))
    assert d == shape.head_dim
    assert block_n % cache.block_size == 0, "BLOCK_N must be a multiple of the page size"

    if out is None:
        out = torch.empty((b, hq, d), dtype=q.dtype, device=q.device)

    kv_code = _KV_CODE[cache.kv_dtype]
    k_scale = cache.k_scale if cache.k_scale is not None else cache.k_cache
    v_scale = cache.v_scale if cache.v_scale is not None else cache.v_cache
    if cache.k_scale is not None:
        stride_sn, stride_st = cache.k_scale.stride()[0], cache.k_scale.stride()[1]
    else:
        stride_sn = stride_st = 0

    if num_splits is None:
        # `.item()` is a full device sync. On Windows WDDM that costs hundreds of
        # microseconds -- more than the kernel itself below batch 8 -- so it must
        # never run when the caller already knows the split count. A serving
        # runtime computes splits once per step, not once per layer.
        max_seq = int(cache.seq_lens.max().item())
        sm = torch.cuda.get_device_properties(q.device).multi_processor_count
        num_splits = pick_num_splits(b, hkv, max_seq, sm, block_n)

    scale = shape.softmax_scale if sm_scale is None else float(sm_scale)
    kb, kt, kh, _ = cache.k_cache.stride()

    if num_splits <= 1:
        _paged_decode_kernel[(b, hkv, 1)](
            q, cache.k_cache, cache.v_cache, k_scale, v_scale,
            out, out, out,  # PartOut/PartLse unused
            cache.block_table, cache.seq_lens,
            scale,
            q.stride(0), q.stride(1),
            kb, kt, kh,
            stride_sn, stride_st,
            out.stride(0), out.stride(1),
            0, 0, 0, 0, 0,
            cache.block_table.stride(0),
            1,
            HEAD_DIM=d, PAGE=cache.block_size, GROUP=group, GROUP_PAD=group_pad,
            BLOCK_N=block_n, KV_DTYPE=kv_code, PER_PAGE_BT=per_page_bt, SPLIT_KV=False,
            num_warps=num_warps, num_stages=num_stages,
        )
        return out

    part_out, part_lse = _WS.get(b, hq, num_splits, d, q.device)
    _paged_decode_kernel[(b, hkv, num_splits)](
        q, cache.k_cache, cache.v_cache, k_scale, v_scale,
        out, part_out, part_lse,
        cache.block_table, cache.seq_lens,
        scale,
        q.stride(0), q.stride(1),
        kb, kt, kh,
        stride_sn, stride_st,
        out.stride(0), out.stride(1),
        part_out.stride(0), part_out.stride(1), part_out.stride(2),
        part_lse.stride(0), part_lse.stride(1),
        cache.block_table.stride(0),
        num_splits,
        HEAD_DIM=d, PAGE=cache.block_size, GROUP=group, GROUP_PAD=group_pad,
        BLOCK_N=block_n, KV_DTYPE=kv_code, PER_PAGE_BT=per_page_bt, SPLIT_KV=True,
        num_warps=num_warps, num_stages=num_stages,
    )
    _split_reduce_kernel[(b, hq)](
        part_out, part_lse, out,
        part_out.stride(0), part_out.stride(1), part_out.stride(2),
        part_lse.stride(0), part_lse.stride(1),
        out.stride(0), out.stride(1),
        num_splits,
        HEAD_DIM=d, SPLIT_PAD=triton.next_power_of_2(num_splits),
        num_warps=4, num_stages=1,
    )
    return out

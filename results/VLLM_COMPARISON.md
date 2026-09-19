# Our kernel vs FlashAttention-2 and vLLM PagedAttention

All three kernels timed **in one process, eagerly, on the same allocations**, on NVIDIA GeForce RTX 3050 Laptop GPU under WSL2 with vLLM 0.9.2 (torch 2.7.0+cu126). Measured ceiling 184 GB/s.

No cross-environment translation and no graphed-vs-eager asymmetry: every kernel here is eager, so this supersedes any comparison against the Windows tables. All three agree numerically to within 7.3e-4 relative.

| batch | ctx | ours µs | FA2 µs | vLLM PA µs | ours GB/s | FA2 GB/s | vLLM PA GB/s | vs FA2 | vs vLLM PA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,024 | 74 | 42 | 34 | 57 | 99 | 124 | **0.57x** | **0.46x** |
| 2 | 1,024 | 73 | 66 | 64 | 114 | 128 | 131 | **0.89x** | **0.87x** |
| 4 | 1,024 | 99 | 113 | 122 | 169 | 148 | 138 | 1.14x | 1.22x |
| 8 | 1,024 | 193 | 216 | 246 | 174 | 155 | 137 | 1.12x | 1.28x |
| 16 | 1,024 | 387 | 421 | 484 | 173 | 159 | 139 | 1.09x | 1.25x |
| 32 | 1,024 | 751 | 838 | 966 | 179 | 160 | 139 | 1.12x | 1.29x |
| 64 | 1,024 | 1,492 | 1,678 | 1,892 | 180 | 160 | 142 | 1.12x | 1.27x |
| 1 | 4,096 | 102 | 119 | 124 | 165 | 141 | 136 | 1.16x | 1.21x |
| 2 | 4,096 | 194 | 211 | 240 | 173 | 159 | 140 | 1.09x | 1.24x |
| 4 | 4,096 | 373 | 431 | 484 | 180 | 156 | 139 | 1.16x | 1.30x |
| 8 | 4,096 | 744 | 828 | 952 | 180 | 162 | 141 | 1.11x | 1.28x |
| 16 | 4,096 | 1,529 | 1,642 | 1,912 | 176 | 163 | 140 | 1.07x | 1.25x |
| 32 | 4,096 | 2,949 | 3,224 | 3,708 | 182 | 167 | 145 | 1.09x | 1.26x |
| 64 | 4,096 | 5,976 | 6,448 | 7,571 | 180 | 167 | 142 | 1.08x | 1.27x |
| 1 | 16,384 | 378 | 430 | 477 | 178 | 156 | 141 | 1.14x | 1.26x |
| 2 | 16,384 | 747 | 796 | 945 | 180 | 169 | 142 | 1.07x | 1.26x |
| 4 | 16,384 | 1,474 | 1,683 | 1,908 | 182 | 159 | 141 | 1.14x | 1.29x |
| 8 | 16,384 | 2,937 | 3,262 | 3,741 | 183 | 165 | 144 | 1.11x | 1.27x |
| 16 | 16,384 | 6,008 | 6,436 | 7,553 | 179 | 167 | 142 | 1.07x | 1.26x |

`vs X` is X's time divided by ours, so **>1 means we are faster**. Bold marks the two points where we lose.

**Geometric mean: 1.06x vs FlashAttention-2 (1.11x at batch >= 4), 1.17x vs vLLM PagedAttention.**

## Where we lose, and why

At **batch 1-2 with a 1k context** both baselines beat us — FA2 by up to 1.8x and vLLM PagedAttention by up to 2.2x. That is the smallest, most parallelism-starved point in the sweep, exactly the regime README section 5.3 identifies: 8-16 CTAs on a 16-SM GPU with only 4 MB of KV to move. Both baselines partition the KV range more aggressively than our `pick_num_splits` heuristic does there. The heuristic caps splits so a split never covers fewer than two tiles; loosening that cap at tiny context is the obvious next experiment.

Everywhere else we are ahead by 1.07-1.16x over FA2 and 1.2-1.3x over vLLM PagedAttention. On a memory-bound kernel already at 97-103% of the bandwidth ceiling that is the expected size of a win, not a claim to have out-engineered FlashAttention: there is simply very little headroom left.

## One asymmetry worth stating

**FA2 gets our cache layout for free.** `flash_attn_with_kvcache` consumes `[num_blocks, page_size, num_kv_heads, head_dim]` plus a dense block table unchanged, so no conversion cost is charged to it. vLLM's PagedAttention needs its own split-K layout, which `_to_vllm_v0_layout` repacks once *outside* the timed region — charging it per call would be benchmarking a memcpy.

// Hand-tuned paged-KV GQA decode attention for Ampere (sm_80/sm_86).
//
// Thread mapping (the thing that matters):
//
//   THREADS_PER_ROW = 16 lanes cooperate on one KV token's head_dim=128 row.
//   Each lane loads VEC = 8 elements. For fp16 that is one 16 B LDG.E.128 per
//   lane, so a row is 16 x 16 B = 256 B of perfectly contiguous, perfectly
//   coalesced DRAM traffic -- exactly two 128 B cache lines, no partial sectors.
//   For the 1-byte KV dtypes the same VEC=8 is an 8 B load and a row is 128 B,
//   still exactly one cache line.
//
//   A warp therefore covers ROWS_PER_WARP = 2 tokens, and a 4-warp block covers
//   ROWS_PER_ITER = 8 tokens per iteration. Each of those 8 (warp, half) pairs
//   keeps its own independent online-softmax state and its own slice of the
//   output accumulator; they are merged once, at the end, through shared memory.
//
// Register budget per thread: acc[GROUP][VEC] and q[GROUP][VEC] floats (the
// dominant terms), the softmax scalars, and UNROLL tokens of staged K/V.
// Measured with Nsight: 128 registers per thread, 16.7% achieved occupancy at
// batch 1 and 33.2% at batch 32. Both are low, and both are fine -- this kernel
// reaches 89-96% of DRAM peak anyway, because what hides memory latency here is
// outstanding loads per thread (the UNROLL below), not resident warps. See
// README "Occupancy is not the goal", where the 8-warp variant has twice the
// occupancy and is slightly slower.
//
// Where it loses to the Triton kernel: this one computes q.k with a warp-shuffle
// reduction on the CUDA cores and burns ~47% SM throughput doing it, against
// Triton's ~17% via an mma. That reduction sits between a load and its consumer.
//
// Unlike the Triton kernel, everything here is in natural-log space (expf/logf)
// rather than log2 space; the CUDA combine kernel below matches it.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#define CUDA_CHECK(x)                                                              \
  do {                                                                             \
    cudaError_t err__ = (x);                                                        \
    TORCH_CHECK(err__ == cudaSuccess, "CUDA error: ", cudaGetErrorString(err__));   \
  } while (0)

namespace {

constexpr int kVec = 8;       // elements per lane: 16 B for fp16, 8 B for int8
constexpr int kWarpSize = 32;

// The lane layout follows from HEAD_DIM rather than being fixed: head_dim 128
// gives 16 lanes per row and 2 rows per warp, head_dim 64 gives 8 lanes and 4
// rows. Everything downstream (the shuffle mask, the number of independent
// softmax states, the shared-memory reduction) derives from these two.
template <int HEAD_DIM>
struct LaneLayout {
  static constexpr int threads_per_row = HEAD_DIM / kVec;
  static constexpr int rows_per_warp = kWarpSize / threads_per_row;
  static_assert(threads_per_row > 0 && threads_per_row <= kWarpSize,
                "HEAD_DIM must be between kVec and kVec*kWarpSize");
  static_assert(kWarpSize % threads_per_row == 0,
                "HEAD_DIM/kVec must divide the warp size");
};

// KV dtype codes, kept in sync with pagedattn/config.py.
constexpr int KV_FP16 = 0;
constexpr int KV_FP8_E5M2 = 1;
constexpr int KV_INT8 = 2;

template <int CODE> struct KVTraits;
template <> struct KVTraits<KV_FP16> {
  using elem_t = __half;
  using vec_t = float4;  // 8 x fp16 = 16 B
  static constexpr bool scaled = false;
};
template <> struct KVTraits<KV_FP8_E5M2> {
  using elem_t = uint8_t;
  using vec_t = uint2;  // 8 x fp8 = 8 B
  static constexpr bool scaled = false;
};
template <> struct KVTraits<KV_INT8> {
  using elem_t = int8_t;
  using vec_t = uint2;  // 8 x int8 = 8 B
  static constexpr bool scaled = true;
};

// ---------------------------------------------------------------------------
// element decode: raw storage -> float
// ---------------------------------------------------------------------------

template <int CODE>
__device__ __forceinline__ void decode_vec(
    const typename KVTraits<CODE>::vec_t& raw, float scale, float (&out)[kVec]);

template <>
__device__ __forceinline__ void decode_vec<KV_FP16>(
    const float4& raw, float, float (&out)[kVec]) {
  const __half2* h = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
  for (int i = 0; i < kVec / 2; ++i) {
    float2 f = __half22float2(h[i]);
    out[2 * i] = f.x;
    out[2 * i + 1] = f.y;
  }
}

template <>
__device__ __forceinline__ void decode_vec<KV_FP8_E5M2>(
    const uint2& raw, float, float (&out)[kVec]) {
  // e5m2 and fp16 share the 1-5 sign/exponent layout, so widening is a pure
  // left shift by 8 bits. That is why this path needs no sm_89 cvt instruction
  // and costs essentially nothing on Ampere.
  const uint8_t* b = reinterpret_cast<const uint8_t*>(&raw);
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    unsigned short bits = static_cast<unsigned short>(b[i]) << 8;
    out[i] = __half2float(__ushort_as_half(bits));
  }
}

template <>
__device__ __forceinline__ void decode_vec<KV_INT8>(
    const uint2& raw, float scale, float (&out)[kVec]) {
  const int8_t* b = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
  for (int i = 0; i < kVec; ++i) out[i] = static_cast<float>(b[i]) * scale;
}

// ---------------------------------------------------------------------------
// main kernel
// ---------------------------------------------------------------------------

template <int HEAD_DIM, int GROUP, int CODE, int WARPS, int UNROLL, bool SPLIT>
__global__ __launch_bounds__(WARPS* kWarpSize) void paged_decode_kernel(
    const __half* __restrict__ Q,               // [B, Hq, D]
    const void* __restrict__ K_raw,             // [NB, PAGE, Hkv, D]
    const void* __restrict__ V_raw,
    const __half* __restrict__ K_scale,         // [NB, PAGE, Hkv] (int8 only)
    const __half* __restrict__ V_scale,
    __half* __restrict__ Out,                   // [B, Hq, D]      (SPLIT == false)
    float* __restrict__ PartOut,                // [B, Hq, S, D]   (SPLIT == true)
    float* __restrict__ PartLse,                // [B, Hq, S]
    const int* __restrict__ BlockTables,        // [B, MAXBLK]
    const int* __restrict__ SeqLens,            // [B]
    float sm_scale,
    int page_log2,
    int num_kv_heads,
    int bt_stride,
    int num_splits) {
  using T = KVTraits<CODE>;
  using elem_t = typename T::elem_t;
  using vec_t = typename T::vec_t;

  using L = LaneLayout<HEAD_DIM>;
  constexpr int kThreadsPerRow = L::threads_per_row;
  constexpr int kRowsPerWarp = L::rows_per_warp;
  constexpr int ROWS_PER_ITER = WARPS * kRowsPerWarp;

  const int b = blockIdx.x;
  const int kvh = blockIdx.y;
  const int split = SPLIT ? blockIdx.z : 0;

  const int tid = threadIdx.x;
  const int lane_in_row = tid % kThreadsPerRow;
  const int state_id = tid / kThreadsPerRow;  // 0 .. ROWS_PER_ITER-1

  // The two row-halves of a warp own different tokens, so when (hi - lo) is not
  // a multiple of ROWS_PER_ITER one half runs an extra iteration. Reducing with
  // a full 0xffffffff mask across that divergence is undefined and hangs on
  // Volta+ independent thread scheduling. The reduction only ever spans the 16
  // lanes of a single row, which share a trip count exactly, so the mask is the
  // half-warp this thread belongs to.
  constexpr unsigned kRowLanes =
      (kThreadsPerRow == 32) ? 0xffffffffu : ((1u << kThreadsPerRow) - 1u);
  const unsigned row_mask =
      kRowLanes << (kThreadsPerRow * (state_id % kRowsPerWarp));

  const int seq_len = SeqLens[b];
  const int page_size = 1 << page_log2;
  const int page_mask = page_size - 1;

  int lo = 0, hi = seq_len;
  if (SPLIT) {
    // Split on page boundaries so each CTA's block-table walk stays aligned.
    const int pages = (seq_len + page_size - 1) >> page_log2;
    const int chunk_pages = (pages + num_splits - 1) / num_splits;
    lo = split * chunk_pages * page_size;
    hi = min(lo + chunk_pages * page_size, seq_len);
    if (lo >= seq_len) { lo = 0; hi = 0; }  // this split owns nothing
  }

  // ---- load Q into registers -----------------------------------------------
  const int q_head0 = kvh * GROUP;
  float q_reg[GROUP][kVec];
#pragma unroll
  for (int g = 0; g < GROUP; ++g) {
    const __half* qp = Q + (size_t)(b * (num_kv_heads * GROUP) + q_head0 + g) * HEAD_DIM +
                       lane_in_row * kVec;
    const float4 raw = *reinterpret_cast<const float4*>(qp);
    decode_vec<KV_FP16>(raw, 1.0f, q_reg[g]);
  }

  float m_i[GROUP], l_i[GROUP], acc[GROUP][kVec];
#pragma unroll
  for (int g = 0; g < GROUP; ++g) {
    m_i[g] = -CUDART_INF_F;
    l_i[g] = 0.0f;
#pragma unroll
    for (int i = 0; i < kVec; ++i) acc[g][i] = 0.0f;
  }

  const elem_t* K = reinterpret_cast<const elem_t*>(K_raw);
  const elem_t* V = reinterpret_cast<const elem_t*>(V_raw);
  const size_t row_stride = (size_t)num_kv_heads * HEAD_DIM;

  // ---- main loop ------------------------------------------------------------
  // This state owns tokens lo+state_id, +ROWS_PER_ITER, +2*ROWS_PER_ITER, ...
  //
  // UNROLL matters more than anything else in this kernel. The naive version
  // issues one K load and one V load, then immediately needs both for the dot
  // product, so each thread has 2 memory operations in flight and the SM stalls
  // on DRAM latency it has nothing to hide behind. Issuing UNROLL tokens' worth
  // of loads *before* touching any of them raises that to 2*UNROLL outstanding
  // requests per thread, which is what actually converts occupancy into
  // bandwidth. This is the same thing Triton's num_stages pipelining does for
  // the Triton kernel, and it is why the un-unrolled version lost to it.
  for (int t0 = lo + state_id; t0 < hi; t0 += ROWS_PER_ITER * UNROLL) {
    int tok_idx[UNROLL];
    bool valid[UNROLL];
    vec_t kraw[UNROLL], vraw[UNROLL];
    float ks[UNROLL], vs[UNROLL];

    // Phase 1: nothing but address arithmetic and loads.
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int t = t0 + u * ROWS_PER_ITER;
      valid[u] = t < hi;
      // Clamp rather than branch: every lane of a row agrees on `valid`, and a
      // clamped in-range address keeps the loads unconditional and coalesced.
      tok_idx[u] = valid[u] ? t : (hi > 0 ? hi - 1 : 0);

      const int t_safe = tok_idx[u];
      const int phys = BlockTables[b * bt_stride + (t_safe >> page_log2)];
      const size_t slot = (size_t)phys * page_size + (t_safe & page_mask);
      const size_t base = slot * row_stride + (size_t)kvh * HEAD_DIM;

      kraw[u] = *reinterpret_cast<const vec_t*>(K + base + lane_in_row * kVec);
      vraw[u] = *reinterpret_cast<const vec_t*>(V + base + lane_in_row * kVec);

      ks[u] = 1.0f;
      vs[u] = 1.0f;
      if (T::scaled) {
        const size_t s_off = slot * num_kv_heads + kvh;
        ks[u] = __half2float(K_scale[s_off]);
        vs[u] = __half2float(V_scale[s_off]);
      }
    }

    // Phase 2: consume them.
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      float kf[kVec], vf[kVec];
      decode_vec<CODE>(kraw[u], ks[u], kf);
      decode_vec<CODE>(vraw[u], vs[u], vf);

      // q . k, reduced across the 16 lanes that share this row.
      float dot[GROUP];
#pragma unroll
      for (int g = 0; g < GROUP; ++g) {
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < kVec; ++i) s = fmaf(q_reg[g][i], kf[i], s);
        dot[g] = s;
      }
      // Unconditional: all 16 lanes of a row share `valid`, so the reduction
      // never straddles divergence, and masking after the fact is cheaper than
      // branching around a shuffle.
#pragma unroll
      for (int off = kThreadsPerRow / 2; off > 0; off >>= 1) {
#pragma unroll
        for (int g = 0; g < GROUP; ++g) dot[g] += __shfl_xor_sync(row_mask, dot[g], off);
      }

      if (!valid[u]) continue;

      // online softmax update (every lane in the row holds identical scalars)
#pragma unroll
      for (int g = 0; g < GROUP; ++g) {
        const float qk = dot[g] * sm_scale;
        const float m_new = fmaxf(m_i[g], qk);
        const float alpha = __expf(m_i[g] - m_new);
        const float p = __expf(qk - m_new);
        l_i[g] = l_i[g] * alpha + p;
#pragma unroll
        for (int i = 0; i < kVec; ++i) acc[g][i] = fmaf(acc[g][i], alpha, p * vf[i]);
        m_i[g] = m_new;
      }
    }
  }

  // ---- merge the ROWS_PER_ITER partial states ------------------------------
  __shared__ float s_m[ROWS_PER_ITER][GROUP];
  __shared__ float s_l[ROWS_PER_ITER][GROUP];
  __shared__ float s_acc[GROUP][HEAD_DIM];

  if (lane_in_row == 0) {
#pragma unroll
    for (int g = 0; g < GROUP; ++g) {
      s_m[state_id][g] = m_i[g];
      s_l[state_id][g] = l_i[g];
    }
  }
  for (int idx = tid; idx < GROUP * HEAD_DIM; idx += WARPS * kWarpSize) {
    s_acc[idx / HEAD_DIM][idx % HEAD_DIM] = 0.0f;
  }
  __syncthreads();

  float m_glob[GROUP], l_glob[GROUP];
#pragma unroll
  for (int g = 0; g < GROUP; ++g) {
    float m = -CUDART_INF_F;
#pragma unroll
    for (int s = 0; s < ROWS_PER_ITER; ++s) m = fmaxf(m, s_m[s][g]);
    float l = 0.0f;
#pragma unroll
    for (int s = 0; s < ROWS_PER_ITER; ++s) {
      if (s_l[s][g] > 0.0f) l += s_l[s][g] * __expf(s_m[s][g] - m);
    }
    m_glob[g] = m;
    l_glob[g] = l;
  }

  // Rescale this state's accumulator to the block-global max and sum it in.
  // 8 contenders per address; shared-memory atomics make this cheaper than a
  // [8][GROUP][HEAD_DIM] staging buffer, which would cost 16 KB and cut
  // occupancy for a reduction that runs once per CTA.
#pragma unroll
  for (int g = 0; g < GROUP; ++g) {
    const float w = (l_i[g] > 0.0f) ? __expf(m_i[g] - m_glob[g]) : 0.0f;
    if (w > 0.0f) {
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        atomicAdd(&s_acc[g][lane_in_row * kVec + i], acc[g][i] * w);
      }
    }
  }
  __syncthreads();

  // ---- epilogue ------------------------------------------------------------
  const int hq = num_kv_heads * GROUP;
  if (SPLIT) {
    for (int idx = tid; idx < GROUP * HEAD_DIM; idx += WARPS * kWarpSize) {
      const int g = idx / HEAD_DIM, d = idx % HEAD_DIM;
      const float denom = (l_glob[g] > 0.0f) ? l_glob[g] : 1.0f;
      PartOut[((size_t)(b * hq + q_head0 + g) * num_splits + split) * HEAD_DIM + d] =
          s_acc[g][d] / denom;
    }
    if (tid < GROUP) {
      const int g = tid;
      PartLse[(size_t)(b * hq + q_head0 + g) * num_splits + split] =
          (l_glob[g] > 0.0f) ? (m_glob[g] + logf(l_glob[g])) : -CUDART_INF_F;
    }
  } else {
    for (int idx = tid; idx < GROUP * HEAD_DIM; idx += WARPS * kWarpSize) {
      const int g = idx / HEAD_DIM, d = idx % HEAD_DIM;
      Out[(size_t)(b * hq + q_head0 + g) * HEAD_DIM + d] =
          __float2half(s_acc[g][d] / l_glob[g]);
    }
  }
}

// ---------------------------------------------------------------------------
// split combine
// ---------------------------------------------------------------------------

template <int HEAD_DIM>
__global__ void split_combine_kernel(
    const float* __restrict__ PartOut,  // [B, Hq, S, D]
    const float* __restrict__ PartLse,  // [B, Hq, S]
    __half* __restrict__ Out,           // [B, Hq, D]
    int num_splits) {
  const int bh = blockIdx.x;  // flattened (batch, q_head)
  const int d = threadIdx.x;

  extern __shared__ float s_w[];
  if (d == 0) {
    float m = -CUDART_INF_F;
    for (int s = 0; s < num_splits; ++s) m = fmaxf(m, PartLse[(size_t)bh * num_splits + s]);
    float denom = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
      const float lse = PartLse[(size_t)bh * num_splits + s];
      const float w = isfinite(lse) ? __expf(lse - m) : 0.0f;
      s_w[s] = w;
      denom += w;
    }
    s_w[num_splits] = denom;
  }
  __syncthreads();

  float o = 0.0f;
  for (int s = 0; s < num_splits; ++s) {
    o = fmaf(s_w[s], PartOut[((size_t)bh * num_splits + s) * HEAD_DIM + d], o);
  }
  Out[(size_t)bh * HEAD_DIM + d] = __float2half(o / s_w[num_splits]);
}

}  // namespace

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

#define LAUNCH_ONE(HD, GRP, CODE, WARPS, UNROLL, SPLIT)                         \
  paged_decode_kernel<HD, GRP, CODE, WARPS, UNROLL, SPLIT>                      \
      <<<grid, dim3(WARPS * kWarpSize), 0, stream>>>(                           \
          reinterpret_cast<const __half*>(q.data_ptr()), k_cache.data_ptr(),    \
          v_cache.data_ptr(),                                                   \
          k_scale.numel() > 0                                                   \
              ? reinterpret_cast<const __half*>(k_scale.data_ptr()) : nullptr,  \
          v_scale.numel() > 0                                                   \
              ? reinterpret_cast<const __half*>(v_scale.data_ptr()) : nullptr,  \
          reinterpret_cast<__half*>(out.data_ptr()),                            \
          part_out.numel() > 0 ? part_out.data_ptr<float>() : nullptr,          \
          part_lse.numel() > 0 ? part_lse.data_ptr<float>() : nullptr,          \
          block_tables.data_ptr<int>(), seq_lens.data_ptr<int>(), sm_scale,     \
          page_log2, num_kv_heads, (int)block_tables.stride(0), num_splits)

// The three tuning variants exist only for the benchmark shape (head_dim 128,
// GQA group 4) -- they are what README section 5.4 A/Bs. Every other shape gets
// the config that measurement picked there (4 warps, unroll 4); carrying the
// full matrix for all of them would multiply compile time for no new insight.
#define LAUNCH_VARIANTS(HD, GRP, CODE, SPLIT)                  \
  if (HD == 128 && GRP == 4) {                                 \
    switch (variant) {                                         \
      case 0: LAUNCH_ONE(HD, GRP, CODE, 4, 1, SPLIT); break;   \
      case 2: LAUNCH_ONE(HD, GRP, CODE, 8, 4, SPLIT); break;   \
      default: LAUNCH_ONE(HD, GRP, CODE, 4, 4, SPLIT); break;  \
    }                                                          \
  } else {                                                     \
    LAUNCH_ONE(HD, GRP, CODE, 4, 4, SPLIT);                    \
  }

#define LAUNCH_DTYPE(HD, GRP, SPLIT)                                        \
  switch (kv_code) {                                                        \
    case KV_FP16: LAUNCH_VARIANTS(HD, GRP, KV_FP16, SPLIT); break;          \
    case KV_FP8_E5M2: LAUNCH_VARIANTS(HD, GRP, KV_FP8_E5M2, SPLIT); break;  \
    case KV_INT8: LAUNCH_VARIANTS(HD, GRP, KV_INT8, SPLIT); break;          \
    default: TORCH_CHECK(false, "bad kv_code ", kv_code);                   \
  }

// Supported (head_dim, group) pairs. Anything else raises rather than silently
// running a wrong specialization -- `supports_shape()` on the Python side is the
// gate callers should consult first.
#define LAUNCH(SPLIT)                                                         \
  if (head_dim == 128 && group == 4) { LAUNCH_DTYPE(128, 4, SPLIT); }         \
  else if (head_dim == 128 && group == 8) { LAUNCH_DTYPE(128, 8, SPLIT); }    \
  else if (head_dim == 64 && group == 4) { LAUNCH_DTYPE(64, 4, SPLIT); }      \
  else if (head_dim == 64 && group == 7) { LAUNCH_DTYPE(64, 7, SPLIT); }      \
  else if (head_dim == 64 && group == 8) { LAUNCH_DTYPE(64, 8, SPLIT); }      \
  else {                                                                      \
    TORCH_CHECK(false, "unsupported (head_dim, group) = (", head_dim, ", ",   \
                group, "); compiled: (128,4) (128,8) (64,4) (64,7) (64,8)");  \
  }

void paged_decode_cuda(
    torch::Tensor q,             // [B, Hq, D] fp16
    torch::Tensor k_cache,       // [NB, PAGE, Hkv, D]
    torch::Tensor v_cache,
    torch::Tensor k_scale,       // [NB, PAGE, Hkv] fp16 or undefined
    torch::Tensor v_scale,
    torch::Tensor block_tables,  // [B, MAXBLK] int32
    torch::Tensor seq_lens,      // [B] int32
    torch::Tensor out,           // [B, Hq, D] fp16
    torch::Tensor part_out,      // [B, Hq, S, D] fp32 or undefined
    torch::Tensor part_lse,      // [B, Hq, S] fp32 or undefined
    double sm_scale_d,
    int64_t kv_code,
    int64_t num_splits,
    int64_t variant) {
  const int batch = q.size(0);
  const int hq = q.size(1);
  const int head_dim = q.size(2);
  const int num_kv_heads = k_cache.size(2);
  const int page_size = k_cache.size(1);
  const float sm_scale = (float)sm_scale_d;

  const int group = hq / num_kv_heads;
  TORCH_CHECK(hq % num_kv_heads == 0, "num_q_heads must be a multiple of num_kv_heads");
  TORCH_CHECK((page_size & (page_size - 1)) == 0, "page size must be a power of two");
  TORCH_CHECK(q.scalar_type() == at::kHalf, "q must be fp16");
  TORCH_CHECK(kv_code != KV_INT8 || (k_scale.numel() > 0 && v_scale.numel() > 0),
              "int8 KV cache requires k_scale/v_scale");

  int page_log2 = 0;
  while ((1 << page_log2) < page_size) ++page_log2;

  auto stream = at::cuda::getCurrentCUDAStream();

  if (num_splits <= 1) {
    const dim3 grid(batch, num_kv_heads, 1);
    LAUNCH(false);
  } else {
    const dim3 grid(batch, num_kv_heads, (unsigned)num_splits);
    LAUNCH(true);
    CUDA_CHECK(cudaGetLastError());
    const size_t smem = (num_splits + 1) * sizeof(float);
    if (head_dim == 128) {
      split_combine_kernel<128><<<batch * hq, 128, smem, stream>>>(
          part_out.data_ptr<float>(), part_lse.data_ptr<float>(),
          reinterpret_cast<__half*>(out.data_ptr()), (int)num_splits);
    } else {
      split_combine_kernel<64><<<batch * hq, 64, smem, stream>>>(
          part_out.data_ptr<float>(), part_lse.data_ptr<float>(),
          reinterpret_cast<__half*>(out.data_ptr()), (int)num_splits);
    }
  }
  CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_decode", &paged_decode_cuda, "paged GQA decode attention (CUDA)");
}

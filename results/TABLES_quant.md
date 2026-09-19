# Decode-attention sweep results

- GPU: **NVIDIA GeForce RTX 3050 Laptop GPU** (sm_86, 16 SMs, 4.0 GiB, driver 561.19)
- PyTorch 2.12.0+cu126 / CUDA 12.6
- Measured streaming-read peak: **180 GB/s** (theoretical 188 GB/s)
- Page size 16, block tables **fragmented**
- Sweep wall time: 19.3 s

## fp16

Latency in microseconds (median), KV dtype `fp16`. `--` = did not run (out of VRAM budget or unsupported shape).

| batch | ctx | Triton (split-KV, per-page BT) | CUDA v2 (4 warps, unroll 4) |
|---:|---:|---:|---:|
| 1 | 4,096 | 98 | 116 |
| 1 | 16,384 | 373 | 393 |
| 8 | 4,096 | 768 | 771 |
| 8 | 16,384 | 3,009 | 2,983 |
| 32 | 4,096 | 2,976 | 3,003 |
| 32 | 16,384 | 11,787 | 11,830 |

Achieved DRAM read bandwidth as a percentage of the measured peak (~180 GB/s streaming read, re-measured once per context block so each row is normalized against the ceiling the GPU had at that moment).

| batch | ctx | Triton (split-KV, per-page BT) | CUDA v2 (4 warps, unroll 4) |
|---:|---:|---:|---:|
| 1 | 4,096 | 95% | 80% |
| 1 | 16,384 | 100% | 95% |
| 8 | 4,096 | 97% | 97% |
| 8 | 16,384 | 99% | 100% |
| 32 | 4,096 | 100% | 99% |
| 32 | 16,384 | 101% | 101% |

## fp8_e5m2

Latency in microseconds (median), KV dtype `fp8_e5m2`. `--` = did not run (out of VRAM budget or unsupported shape).

| batch | ctx | Triton (split-KV, per-page BT) | CUDA v2 (4 warps, unroll 4) |
|---:|---:|---:|---:|
| 1 | 4,096 | 54 | 82 |
| 1 | 16,384 | 195 | 266 |
| 8 | 4,096 | 381 | 435 |
| 8 | 16,384 | 1,521 | 1,749 |
| 32 | 4,096 | 1,540 | 1,736 |
| 32 | 16,384 | 6,045 | 6,703 |

Achieved DRAM read bandwidth as a percentage of the measured peak (~180 GB/s streaming read, re-measured once per context block so each row is normalized against the ceiling the GPU had at that moment).

| batch | ctx | Triton (split-KV, per-page BT) | CUDA v2 (4 warps, unroll 4) |
|---:|---:|---:|---:|
| 1 | 4,096 | 86% | 57% |
| 1 | 16,384 | 96% | 70% |
| 8 | 4,096 | 98% | 86% |
| 8 | 16,384 | 98% | 85% |
| 32 | 4,096 | 97% | 86% |
| 32 | 16,384 | 99% | 89% |

## int8

Latency in microseconds (median), KV dtype `int8`. `--` = did not run (out of VRAM budget or unsupported shape).

| batch | ctx | Triton (split-KV, per-page BT) | CUDA v2 (4 warps, unroll 4) |
|---:|---:|---:|---:|
| 1 | 4,096 | 75 | 94 |
| 1 | 16,384 | 281 | 321 |
| 8 | 4,096 | 556 | 519 |
| 8 | 16,384 | 2,236 | 2,075 |
| 32 | 4,096 | 1,991 | 2,048 |
| 32 | 16,384 | 7,965 | 8,164 |

Achieved DRAM read bandwidth as a percentage of the measured peak (~180 GB/s streaming read, re-measured once per context block so each row is normalized against the ceiling the GPU had at that moment).

| batch | ctx | Triton (split-KV, per-page BT) | CUDA v2 (4 warps, unroll 4) |
|---:|---:|---:|---:|
| 1 | 4,096 | 63% | 50% |
| 1 | 16,384 | 67% | 59% |
| 8 | 4,096 | 68% | 73% |
| 8 | 16,384 | 68% | 73% |
| 32 | 4,096 | 76% | 74% |
| 32 | 16,384 | 76% | 74% |

## Speedup vs the PyTorch baselines

Speedup of `Triton (split-KV, per-page BT)` over the PyTorch baselines, and its absolute decode throughput.

| batch | ctx | vs SDPA mem-efficient | vs SDPA math | tokens/s (ours) |
|---:|---:|---:|---:|---:|
| 1 | 4,096 | -- | -- | 10,157 |
| 1 | 16,384 | -- | -- | 2,684 |
| 8 | 4,096 | -- | -- | 10,420 |
| 8 | 16,384 | -- | -- | 2,659 |
| 32 | 4,096 | -- | -- | 10,754 |
| 32 | 16,384 | -- | -- | 2,715 |

⚠ = the baseline is thrashing under memory pressure at this point; the ratio is not a meaningful kernel comparison.

## Launch overhead (eager vs CUDA graph)

Host-side launch overhead for `Triton (split-KV, per-page BT)`: the same call measured eagerly and under CUDA-graph replay. The gap is pure host cost and is what a serving runtime removes by capturing the decode step.

| batch | ctx | eager us | graphed us | overhead us | overhead share |
|---:|---:|---:|---:|---:|---:|
| 1 | 4,096 | 238 | 98 | 140 | 59% |
| 1 | 16,384 | 416 | 373 | 43 | 10% |
| 8 | 4,096 | 828 | 768 | 60 | 7% |
| 8 | 16,384 | 3,188 | 3,009 | 180 | 6% |
| 32 | 4,096 | 3,132 | 2,976 | 157 | 5% |
| 32 | 16,384 | 11,997 | 11,787 | 209 | 2% |

## KV-cache quantization

KV-cache quantization, `triton` implementation. `GB/s` is the achieved read bandwidth over the *stored* bytes, so a 1-byte cache moving the same tokens in half the time shows the same GB/s at half the latency.

| batch | ctx | fp16 us | fp8_e5m2 us | int8 us | int8 vs fp16 |
|---:|---:|---:|---:|---:|---:|
| 1 | 4,096 | 98 | 54 | 75 | 1.32x |
| 1 | 16,384 | 373 | 195 | 281 | 1.32x |
| 8 | 4,096 | 768 | 381 | 556 | 1.38x |
| 8 | 16,384 | 3,009 | 1,521 | 2,236 | 1.35x |
| 32 | 4,096 | 2,976 | 1,540 | 1,991 | 1.49x |
| 32 | 16,384 | 11,787 | 6,045 | 7,965 | 1.48x |

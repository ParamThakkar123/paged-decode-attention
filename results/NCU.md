# Nsight Compute summary

Averaged over the profiled launches of each kernel. `sectors/req` is sectors per global load request. These kernels load 128 bits per lane, so a warp covers 32 x 16 B = 512 B = **16** sectors: 16 is perfect coalescing here, and 32 would mean every lane fetched its own sector.

| config | kernel | time | DRAM %peak | SM %peak | DRAM read | occupancy | sectors/req | grid | block | regs | tensor % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cuda-b1-s16384 | `void <unnamed>::split_combine_kernel<128>` | 3.81 | 14.14 | 5.16 | 19.60 | 16.31 | 3.00 | 32.00 | 128 | 42.00 | 0.00 |
| cuda-b1-s16384 | `void <unnamed>::paged_decode_kernel<128, 4, ` | 403 | 89.73 | 46.80 | 167 | 16.74 | 10.99 | 32.00 | 128 | 128 | 0.00 |
| cuda-b32-s4096 | `void <unnamed>::paged_decode_kernel<128, 4, ` | 3.00 | 95.59 | 49.45 | 179 | 33.18 | 10.99 | 256 | 128 | 128 | 0.00 |
| triton-b1-s16384-nosplit | `_paged_decode_kernel` | 682 | 53.00 | 9.37 | 98.40 | 8.36 | 15.77 | 8.00 | 128 | 80.00 | 15.63 |
| triton-b1-s16384-pertokbt | `_split_reduce_kernel` | 3.52 | 13.86 | 10.07 | 20.17 | 17.83 | 6.00 | 32.00 | 128 | 23.00 | 0.00 |
| triton-b1-s16384-pertokbt | `_paged_decode_kernel` | 377 | 95.96 | 19.33 | 178 | 16.87 | 15.53 | 32.00 | 128 | 80.00 | 14.41 |
| triton-b1-s16384 | `_split_reduce_kernel` | 3.42 | 13.88 | 10.16 | 20.73 | 16.81 | 6.00 | 32.00 | 128 | 23.00 | 0.00 |
| triton-b1-s16384 | `_paged_decode_kernel` | 379 | 95.39 | 17.27 | 177 | 16.79 | 15.76 | 32.00 | 128 | 80.00 | 14.37 |
| triton-b32-s4096 | `_paged_decode_kernel` | 2.94 | 97.57 | 17.51 | 183 | 16.65 | 15.76 | 256 | 128 | 80.00 | 14.49 |

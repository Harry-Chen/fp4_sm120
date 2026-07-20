# Random Hadamard Transform GEMM for SM120

SM120 port of the Random Hadamard Transform (RHT) GEMM kernel used by
Transformer Engine's FP4 training recipe.

## Overview

The FP4 training recipe applies a random Hadamard rotation to activations and
weights before FP4 quantization. The existing Transformer Engine kernel
(`rht_gemm_ntt_w_sfc` in `hadamard_transform_cast_fusion.cu`) uses SM100-only
features (UMMA/tcgen05, TMEM, 232KB shared memory) that are not available on
SM120 family GPUs (RTX 50x0).

This implementation provides a drop-in replacement using:

- **WMMA** (`wmma.mma.sync.aligned.m16n16k16.f32.bf16.bf16.f32`) for the 16x16
  Hadamard matrix multiply — the natural fit since B is always exactly 16x16.
- **PTX FP4/FP8 conversion** (`cvt.rn.satfinite.e2m1x2.f32`,
  `cvt.rn.satfinite.e4m3x2.f32`) for quantization.
- **Software stochastic rounding** (noise injection + round-to-nearest) from
  the `stochastic_rounding/` polyfill since `cvt.rs.satfinite.e2m1x4.f32` is
  not available on SM120.

Shared memory usage is ~18KB per block (smem_A 4352 B + smem_B 512 B +
smem_result 8192 B + smem_packed 4096 B + smem_sfc 512 B), well within
SM120's 99KB limit.

## Operation

```
Input:  A (m × n, BF16, col-major)  ×  B (16 × 16, BF16, Hadamard matrix)
Output: C (m × n, FP4 E2M1, row-major)  +  SFC (m × n/16, FP8 UE4M3, row-major)
```

For each group of 16 columns in A, the kernel:
1. Multiplies by the 16×16 Hadamard matrix B via WMMA
2. Optionally rounds through BF16 for bitwise compatibility (`!kUseFastMath`)
3. Computes per-16-element amax and FP8 UE4M3 scale factor (SFC)
4. Quantizes to FP4 E2M1 with optional stochastic rounding

## API

```cpp
#include "rht_gemm_sm120.cuh"

rht_gemm_sm120::rht_gemm_ntt_w_sfc<TA, TB, TC, TSFC,
    kEnableStochasticRounding, kUseFastMath>(
    m, n, A, B, C, SFC, global_amax, rng_state, sm_count, stream, k_tile_size);
```

Same signature as the reference `rht_gemm_ntt_w_sfc`. Constraints: `m % 128 == 0`
and `n % 64 == 0`.

## Building

### Correctness test + benchmark (SM120)

Runs on RTX 5090 or any SM120 family GPU. Compares the kernel output against
a naive CPU reference.

```bash
make                        # default CUDA_ARCH=120a
make CUDA_ARCH=120f         # family-compatible
make CUDA_ARCH=121a         # SM121 specific
./test_correctness.exe
```

Requires CUDA 13.1+ and CUTLASS headers at `$CUTLASS_HOME` (default
`/home/harry/cutlass`). Header-only — no CUTLASS runtime dependency.

### Comparison test vs TE reference (SM100)

Compiles both our WMMA kernel and the original Transformer Engine SM100
UMMA kernel into a single binary. Must be **run on an SM100 GPU** (GB200,
B200, etc.) to execute both kernels and compare outputs byte-by-byte.

```bash
make test_compare.exe       # compiles for compute_100a
./test_compare.exe          # run on GB200
```

Additional requirements:

- **CUTLASS** source at `$CUTLASS_HOME` (default `/home/harry/cutlass`)
- **Transformer Engine** source at `$TE_HOME` (default
  `/home/harry/TransformerEngine`) — only the header files are used
- The TE headers with heavy dependencies (cuDNN, etc.) are replaced by
  minimal stubs in `te_ref/stubs/`; the real TE `ptx.cuh` and
  `curanddx.hpp` are symlinked from the TE source tree

The comparison test uses separate compilation (`-rdc=true`) to isolate our
kernel's includes (`sr.sm120.cuh`) from the TE reference's includes
(`ptx.cuh`) which define overlapping symbols.

## Performance

On RTX 5090 (1792 GB/s memory bandwidth), measured with `bench_dram.cu`
(each iteration reads a fresh buffer to defeat L2 caching, isolating true
DRAM bandwidth):

| Size (m × n) | Baseline (GB/s) | Optimized (GB/s) | Speedup | % of Peak |
|---|---|---|---|---|
| 1024 × 1024 | 418 | 434 | +4% | 24% |
| 2048 × 2048 | 871 | 1304 | +50% | 73% |
| 4096 × 4096 | 1162 | 1742 | +50% | 97% |
| 8192 × 5120 | 1247 | 1867 | +50% | 104% |
| 8192 × 10240 | 1264 | 1568 | +24% | 88% |

The kernel is memory-bound (arithmetic intensity ~12.7 FLOP/byte). The
remaining gap to peak at large sizes is primarily due to the row-major FP4
output layout: within a warp, 4 consecutive rows are written per store
cycle, each 32 bytes wide but N/2 bytes apart, causing 4× sector
amplification on global stores. At smaller sizes (≤ 4096), the working
set fits partially in L2, so measured bandwidth can exceed DRAM peak.

### Optimizations

The following optimizations were applied to the kernel:

1. **Coalesced FP4 writeback via smem staging buffer** — warp results are
   first written to a 128×64 byte `smem_packed` buffer in shared memory,
   then all 256 threads perform coalesced `uint4` stores to global memory
   (8 threads per 32-byte row). Eliminates the scattered 4-byte writes
   that previously caused 8× sector amplification.

2. **Coalesced SFC writeback via smem staging** — the 1-byte per-row SFC
   (scale factor) writes are staged to a 128×4 byte `smem_sfc` buffer,
   then written as a single `uint32` per row. Eliminates 16× sector
   amplification from strided 1-byte stores.

3. **smem_A stride padding (128 → 136)** — the col-major smem_A buffer is
   padded from stride 128 to 136 elements. This shifts each column's bank
   assignment, reducing the 8-way bank conflict on the WMMA
   `load_matrix_sync` to 2-way. Stride 136 is a multiple of 8, preserving
   `uint4` store alignment.

4. **Register-based byte extraction** — the partial-tail write path
   extracts bytes from `uint4` via register shifts instead of
   `reinterpret_cast` indexing, eliminating local memory spilling.

## Files

- `rht_gemm_sm120.cuh` — Main kernel header (drop-in replacement)
- `sr.sm120.cuh` — Symlink to `../stochastic_rounding/sr.sm120.cuh` (FP4 SR polyfill)
- `test_correctness.cu` — Correctness test vs naive CPU reference + benchmark
- `bench_rht_only.cu` — Standalone RHT benchmark for ncu/nsys profiling
- `bench_dram.cu` — DRAM bandwidth benchmark (rotates A buffers to defeat L2)
- `test_compare.cu` — Main driver for SM100 comparison test
- `test_compare_ours.cu` — Our kernel instantiation (separate TU)
- `test_compare_ref.cu` — TE reference kernel instantiation (separate TU)
- `te_ref/` — Original TE reference implementation (SM100 only)
  - `hadamard_transform_cast_fusion.cu` — Full TE source (for reference)
  - `hadamard_transform_cast_fusion_core.inc` — Extracted core kernel code
  - `stubs/` — Minimal header stubs replacing TE's heavy dependencies

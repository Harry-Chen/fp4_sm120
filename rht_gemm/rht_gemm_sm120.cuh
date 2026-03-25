#pragma once

// RHT GEMM kernel for SM120 family GPUs (RTX 50x0)
// Drop-in replacement for rht_gemm_ntt_w_sfc from Transformer Engine's
// hadamard_transform_cast_fusion.cu.
//
// Uses WMMA (wmma.mma.sync m16n16k16 BF16) instead of SM100 UMMA (tcgen05).
// Shared memory usage: ~13KB per block (well within SM120's 99KB limit).
//
// Operation: For each group of 16 columns in A, multiply by the 16x16
// Hadamard matrix B, then quantize the result to FP4 E2M1 with per-block
// FP8 UE4M3 scale factors (SFC).

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstdio>

#include "sr.sm120.cuh"  // cvt_e2m1x4_rn, fp32x4_to_e2m1x4_sr, apply_sr_noise_e2m1

namespace rht_gemm_sm120 {

// ============================================================
// Kernel tuning parameters
// ============================================================
static constexpr int TILE_M = 128;
static constexpr int TILE_N = 16;
static constexpr int GROUPS_PER_BLOCK = 4;
static constexpr int TILE_N_BLOCK = TILE_N * GROUPS_PER_BLOCK;  // 64
static constexpr int WARPS_PER_BLOCK = 8;
static constexpr int WARP_SIZE = 32;
static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

static constexpr float FP4_MAX = 6.0f;
static constexpr float FP8_E4M3_MAX = 448.0f;

// ============================================================
// NaN-propagating min/max (matching CUTLASS minimum/maximum_with_nan_propagation)
// CUDA's fminf/fmaxf follow IEEE 754-2008 which treats NaN as "missing" —
// the reference kernel uses NaN-propagating variants so we must too.
// ============================================================

__device__ __forceinline__
float fmax_nan_prop(float a, float b) {
    float r;
    asm("max.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__
float fmin_nan_prop(float a, float b) {
    float r;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

// ============================================================
// FP8 UE4M3 conversion helpers (unsigned E4M3, range [0, 448])
// ============================================================

__device__ __forceinline__
uint8_t float_to_ue4m3(float val) {
    uint16_t tmp;
    float zero = 0.0f;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                 : "=h"(tmp) : "f"(zero), "f"(val));
    return static_cast<uint8_t>(tmp & 0xFF);
}

__device__ __forceinline__
float ue4m3_to_float(uint8_t val) {
    uint16_t bits = static_cast<uint16_t>(val);
    uint32_t packed;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(packed) : "h"(bits));
    return __half2float(reinterpret_cast<__half2 const&>(packed).x);
}

// ============================================================
// Philox4x32-10 RNG — matches TE's curanddx implementation.
// Counter layout: ctr = {offset_lo, offset_hi, subseq_lo, subseq_hi}
// with carry propagation on skip_offset / skip_subsequence.
// ============================================================

struct Philox4x32 {
    uint4 ctr;
    uint2 key;

    __device__ void init(uint64_t seed, uint64_t subsequence, uint64_t offset) {
        ctr = make_uint4(0, 0, 0, 0);
        key.x = static_cast<uint32_t>(seed);
        key.y = static_cast<uint32_t>(seed >> 32);
        // skip_subsequence: increment ctr.z/w
        uint32_t nlo = static_cast<uint32_t>(subsequence);
        uint32_t nhi = static_cast<uint32_t>(subsequence >> 32);
        ctr.z += nlo;
        if (ctr.z < nlo) nhi++;
        ctr.w += nhi;
        // skip_offset: increment ctr.x/y
        nlo = static_cast<uint32_t>(offset);
        nhi = static_cast<uint32_t>(offset >> 32);
        ctr.x += nlo;
        if (ctr.x < nlo) nhi++;
        ctr.y += nhi;
    }

    __device__ uint4 generate() {
        uint4 c = ctr;
        uint2 k = key;
        #pragma unroll
        for (int i = 0; i < 9; i++) {
            c = single_round(c, k);
            k.x += 0x9E3779B9u;
            k.y += 0xBB67AE85u;
        }
        c = single_round(c, k);
        // Increment counter with carry
        if (++ctr.x == 0) if (++ctr.y == 0) if (++ctr.z == 0) ++ctr.w;
        return c;
    }

    __device__ static uint4 single_round(uint4 c, uint2 k) {
        uint32_t hi0 = __umulhi(0xD2511F53u, c.x);
        uint32_t lo0 = 0xD2511F53u * c.x;
        uint32_t hi1 = __umulhi(0xCD9E8D57u, c.z);
        uint32_t lo1 = 0xCD9E8D57u * c.z;
        return make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    }
};

// ============================================================
// Global encode scale for FP4 quantization
// ============================================================

__device__ __forceinline__
float compute_global_encode_scale(float global_amax) {
    float scale = FP8_E4M3_MAX * FP4_MAX / global_amax;
    scale = fminf(scale, FLT_MAX);
    return (global_amax == 0.f || scale == 0.f) ? 1.f : scale;
}

// ============================================================
// Main RHT GEMM kernel
//
// Each block processes TILE_M (128) rows x GROUPS_PER_BLOCK (4) x
// TILE_N (16) columns = 128 x 64 output tile.
// 8 warps, each handles 16x16 via WMMA per column group.
// ============================================================

template <bool kEnableStochasticRounding, bool kUseFastMath>
__global__ void
rht_gemm_kernel(
    int M, int N,
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    uint8_t* __restrict__ C,
    uint8_t* __restrict__ SFC,
    const float* __restrict__ global_amax,
    const size_t* __restrict__ rng_state)
{
    using namespace nvcuda::wmma;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int block_row = blockIdx.x * TILE_M;
    const int block_col_base = blockIdx.y * TILE_N_BLOCK;

    // --- Shared memory layout ---
    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_A = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_B = smem_A + TILE_M * TILE_N;
    float* smem_result = reinterpret_cast<float*>(smem_B + TILE_N * TILE_N);

    // --- Load B (16x16, row-major) once ---
    for (int i = threadIdx.x; i < TILE_N * TILE_N; i += THREADS_PER_BLOCK) {
        smem_B[i] = B[i];
    }

    // Precompute quantization constants
    const float global_amax_val = *global_amax;
    const float global_encode_scale = compute_global_encode_scale(global_amax_val);
    const float global_decode_scale = 1.0f / global_encode_scale;
    const float scale_multiplier = global_encode_scale / FP4_MAX;

    // Preload B fragment (reused across all column groups)
    __syncthreads();
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_frag;
    load_matrix_sync(b_frag, smem_B, TILE_N);

    const int warp_row_start = warp_id * 16;

    // --- Process GROUPS_PER_BLOCK column groups ---
    for (int g = 0; g < GROUPS_PER_BLOCK; g++) {
        const int col_start = block_col_base + g * TILE_N;
        if (col_start >= N) break;
        const int col_group_idx = col_start / TILE_N;

        // Load A tile (128x16, col-major) into shared memory
        {
            constexpr int ELEMS_PER_THREAD = 8;
            constexpr int THREADS_PER_COL = TILE_M / ELEMS_PER_THREAD;
            const int col = threadIdx.x / THREADS_PER_COL;
            const int row_base = (threadIdx.x % THREADS_PER_COL) * ELEMS_PER_THREAD;

            if (col < TILE_N) {
                const __nv_bfloat16* src = A + (block_row + row_base)
                                         + (long long)M * (col_start + col);
                __nv_bfloat16* dst = smem_A + col * TILE_M + row_base;
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
            }
        }
        __syncthreads();

        // WMMA: each warp computes 16x16 Hadamard transform
        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, col_major> a_frag;
        fragment<accumulator, 16, 16, 16, float> c_frag;

        load_matrix_sync(a_frag, smem_A + warp_row_start, TILE_M);
        fill_fragment(c_frag, 0.0f);
        mma_sync(c_frag, a_frag, b_frag, c_frag);

        // Store WMMA result to shared memory (row-major)
        float* warp_result = smem_result + warp_id * 16 * 16;
        store_matrix_sync(warp_result, c_frag, 16, mem_row_major);

        // --- Quantize: FP32 → FP4 with per-16-element SFC ---
        const int row_in_tile = lane_id / 2;
        const int half = lane_id % 2;
        const int global_row = block_row + warp_row_start + row_in_tile;

        if (row_in_tile < 16 && global_row < M) {
            float vals[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                vals[i] = warp_result[row_in_tile * 16 + half * 8 + i];
            }

            if constexpr (!kUseFastMath) {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    vals[i] = __bfloat162float(__float2bfloat16(vals[i]));
                }
            }

            // Amax with NaN propagation (matching reference's
            // cutlass::maximum_absolute_value_reduction<..., true>).
            // Use fast fmaxf in the hot loop; detect NaN via per-element
            // flag OR (compiled to predicated set + OR, no branch).
            float local_max = 0.0f;
            int has_nan = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                has_nan |= __isnanf(vals[i]);
                local_max = fmaxf(local_max, fabsf(vals[i]));
            }
            has_nan |= __shfl_xor_sync(0xFFFFFFFF, has_nan, 1);

            float other_max = __shfl_xor_sync(0xFFFFFFFF, local_max, 1);
            float row_max = fmaxf(local_max, other_max);
            if (has_nan) row_max = NAN;

            float pvscale = row_max * scale_multiplier;
            uint8_t pvscale_fp8 = float_to_ue4m3(pvscale);
            float pvscale_dequant = ue4m3_to_float(pvscale_fp8);
            float qpvscale_scaled = pvscale_dequant * global_decode_scale;
            float acc_scale;
            if constexpr (kUseFastMath) {
                acc_scale = __frcp_rn(qpvscale_scaled);
            } else {
                // Reference uses cutlass::divides (IEEE 754: 1/0 → Inf)
                acc_scale = 1.0f / qpvscale_scaled;
            }
            // NaN-propagating clamp (matching reference's minimum_with_nan_propagation)
            acc_scale = fmin_nan_prop(acc_scale, FLT_MAX);

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                vals[i] *= acc_scale;
            }

            // Pack to FP4 (reversed arg order for correct nibble layout)
            uint16_t packed_lo, packed_hi;
            if constexpr (kEnableStochasticRounding) {
                const uint64_t rng_seed   = rng_state ? rng_state[0] : 0;
                const uint64_t rng_offset = rng_state ? rng_state[1] : 0;
                const uint64_t rng_seq    = threadIdx.x
                    + (uint64_t)blockIdx.x * blockDim.x
                    + (uint64_t)blockIdx.y * blockDim.x * gridDim.x
                    + (uint64_t)g * blockDim.x * gridDim.x * gridDim.y;
                Philox4x32 rng;
                rng.init(rng_seed, rng_seq, rng_offset);
                uint4 rand = rng.generate();
                packed_lo = ::fp32x4_to_e2m1x4_sr(vals[3], vals[2], vals[1], vals[0], rand.x);
                packed_hi = ::fp32x4_to_e2m1x4_sr(vals[7], vals[6], vals[5], vals[4], rand.y);
            } else {
                packed_lo = ::cvt_e2m1x4_rn(vals[3], vals[2], vals[1], vals[0]);
                packed_hi = ::cvt_e2m1x4_rn(vals[7], vals[6], vals[5], vals[4]);
            }
            uint32_t packed = static_cast<uint32_t>(packed_lo)
                            | (static_cast<uint32_t>(packed_hi) << 16);

            int byte_offset = (global_row * N + col_start + half * 8) / 2;
            *reinterpret_cast<uint32_t*>(C + byte_offset) = packed;

            if (half == 0) {
                SFC[global_row * (N / TILE_N) + col_group_idx] = pvscale_fp8;
            }
        }
        __syncthreads();  // Before next group overwrites smem_A
    }
}

// ============================================================
// Host launcher — drop-in replacement for rht_gemm_ntt_w_sfc
// ============================================================

template <typename TA, typename TB, typename TC, typename TSFC,
          bool kEnableStochasticRounding = false,
          bool kUseFastMath = false>
void rht_gemm_ntt_w_sfc(
    int m, int n,
    TA const* A,
    TB const* B,
    TC* C,
    TSFC* SFC,
    float const* global_amax,
    const size_t* rng_state,
    uint32_t sm_count,
    cudaStream_t stream,
    int k_tile_size = 2048)
{
    if (m == 0 || n == 0) return;

    assert(m % TILE_M == 0 && "M must be a multiple of 128");
    assert(n % (4 * TILE_N) == 0 && "N must be a multiple of 64");

    dim3 grid(m / TILE_M, (n + TILE_N_BLOCK - 1) / TILE_N_BLOCK);
    dim3 block(THREADS_PER_BLOCK);

    int smem_size = TILE_M * TILE_N * sizeof(__nv_bfloat16)
                  + TILE_N * TILE_N * sizeof(__nv_bfloat16)
                  + TILE_M * TILE_N * sizeof(float);

    auto kernel = &rht_gemm_kernel<kEnableStochasticRounding, kUseFastMath>;

    cudaFuncSetAttribute(kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);

    kernel<<<grid, block, smem_size, stream>>>(
        m, n,
        reinterpret_cast<const __nv_bfloat16*>(A),
        reinterpret_cast<const __nv_bfloat16*>(B),
        reinterpret_cast<uint8_t*>(C),
        reinterpret_cast<uint8_t*>(SFC),
        global_amax, rng_state);
}

}  // namespace rht_gemm_sm120

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

namespace rht_gemm_sm120 {

// ============================================================
// Kernel tuning parameters
// ============================================================
static constexpr int TILE_M = 128;
static constexpr int TILE_N = 16;
static constexpr int WARPS_PER_BLOCK = 8;
static constexpr int WARP_SIZE = 32;
static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

// NVFP4 quantization constants
static constexpr float FP4_MAX = 6.0f;
static constexpr float FP8_E4M3_MAX = 448.0f;

// ============================================================
// FP8 UE4M3 conversion helpers (unsigned E4M3, range [0, 448])
// Uses cvt.rn.satfinite.e4m3x2.f32 PTX instruction.
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
// FP4 E2M1 conversion (round-to-nearest)
// Packs 4 FP32 values into 16-bit (4 nibbles).
// Layout: bits[3:0]=d, [7:4]=c, [11:8]=b, [15:12]=a
// ============================================================

__device__ __forceinline__
uint16_t cvt_e2m1x4_rn(float a, float b, float c, float d) {
    uint16_t result;
    asm volatile(
        "{\n\t"
        ".reg .b8  lo8, hi8;\n\t"
        ".reg .b32 lo32, hi32, packed32;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 lo8, %3, %4;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 hi8, %1, %2;\n\t"
        "cvt.u32.u8 lo32, lo8;\n\t"
        "cvt.u32.u8 hi32, hi8;\n\t"
        "shl.b32 hi32, hi32, 8;\n\t"
        "or.b32  packed32, lo32, hi32;\n\t"
        "cvt.u16.u32 %0, packed32;\n\t"
        "}"
        : "=h"(result)
        : "f"(a), "f"(b), "f"(c), "f"(d)
    );
    return result;
}

// ============================================================
// Stochastic rounding noise injection for E2M1
// From sr.sm120.cuh: adds noise in [-ULP/2, ULP/2) then clamps,
// so that hardware RN yields correct SR probability.
// ============================================================

__device__ __forceinline__
float apply_sr_noise_e2m1(float x, unsigned rand_byte) {
    unsigned u = __float_as_uint(x);
    unsigned abs_u = u & 0x7FFFFFFFu;
    unsigned exp = abs_u >> 23;

    unsigned ulp_bexp = min(max(exp, 127u), 129u) - 1u;
    float ulp = __uint_as_float(ulp_bexp << 23);

    float x_lo_abs;
    if (exp >= 127u) {
        x_lo_abs = __uint_as_float(abs_u & 0xFFC00000u);
    } else {
        x_lo_abs = (exp >= 126u) ? 0.5f : 0.0f;
    }

    float noise = (float)((int)rand_byte - 127.5f) * __uint_as_float(0x3B800000u) * ulp;
    float ax_noisy = fmaxf(fabsf(x) + noise, x_lo_abs);
    return copysignf(ax_noisy, x);
}

__device__ __forceinline__
uint16_t cvt_e2m1x4_sr(float a, float b, float c, float d, uint32_t rbits) {
    a = apply_sr_noise_e2m1(a, (rbits      ) & 0xFFu);
    b = apply_sr_noise_e2m1(b, (rbits >>  8) & 0xFFu);
    c = apply_sr_noise_e2m1(c, (rbits >> 16) & 0xFFu);
    d = apply_sr_noise_e2m1(d, (rbits >> 24) & 0xFFu);
    return cvt_e2m1x4_rn(a, b, c, d);
}

// ============================================================
// Philox4x32-10 RNG (compatible with Transformer Engine)
// ============================================================

struct Philox4x32 {
    uint32_t ctr[4];
    uint32_t key[2];

    __device__ void init(uint64_t seed, uint64_t sequence, uint64_t offset) {
        ctr[0] = static_cast<uint32_t>(sequence);
        ctr[1] = static_cast<uint32_t>(sequence >> 32);
        ctr[2] = static_cast<uint32_t>(offset);
        ctr[3] = static_cast<uint32_t>(offset >> 32);
        key[0] = static_cast<uint32_t>(seed);
        key[1] = static_cast<uint32_t>(seed >> 32);
    }

    __device__ uint4 generate() {
        uint32_t c[4] = {ctr[0], ctr[1], ctr[2], ctr[3]};
        uint32_t k[2] = {key[0], key[1]};
        for (int i = 0; i < 10; i++) {
            round(c, k);
            k[0] += 0x9E3779B9u;
            k[1] += 0xBB67AE85u;
        }
        ctr[0]++;
        if (ctr[0] == 0) ctr[1]++;
        return {c[0], c[1], c[2], c[3]};
    }

    __device__ static void round(uint32_t* c, const uint32_t* k) {
        uint64_t r0 = static_cast<uint64_t>(0xD2511F53u) * c[0];
        uint64_t r1 = static_cast<uint64_t>(0xCD9E8D57u) * c[2];
        uint32_t hi0 = static_cast<uint32_t>(r0 >> 32);
        uint32_t lo0 = static_cast<uint32_t>(r0);
        uint32_t hi1 = static_cast<uint32_t>(r1 >> 32);
        uint32_t lo1 = static_cast<uint32_t>(r1);
        c[0] = hi1 ^ c[1] ^ k[0];
        c[1] = lo1;
        c[2] = hi0 ^ c[3] ^ k[1];
        c[3] = lo0;
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
// Each block processes TILE_M (128) rows x TILE_N (16) columns.
// 8 warps, each handles 16x16 via WMMA.
// ============================================================

template <bool kEnableStochasticRounding, bool kUseFastMath>
__global__ void rht_gemm_kernel(
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
    const int col_group = blockIdx.y;
    const int col_start = col_group * TILE_N;

    // --- Shared memory layout ---
    // smem_A: 128 x 16 BF16, col-major (4096 bytes)
    // smem_B: 16 x 16 BF16, row-major (512 bytes)
    // smem_C: 128 x 16 FP32, per-warp row-major (8192 bytes)
    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_A = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_B = smem_A + TILE_M * TILE_N;
    float* smem_C = reinterpret_cast<float*>(smem_B + TILE_N * TILE_N);

    // --- Load B (16x16, row-major) into shared memory ---
    for (int i = threadIdx.x; i < TILE_N * TILE_N; i += THREADS_PER_BLOCK) {
        smem_B[i] = B[i];
    }

    // --- Load A tile (128x16, col-major) into shared memory ---
    // smem_A layout: smem_A[col * 128 + row], matching A's col-major stride
    // Vectorized: each thread loads 8 BF16 (16 bytes) contiguously within a column
    {
        constexpr int ELEMS_PER_THREAD = 8;
        constexpr int THREADS_PER_COL = TILE_M / ELEMS_PER_THREAD;  // 16
        const int col = threadIdx.x / THREADS_PER_COL;
        const int row_base = (threadIdx.x % THREADS_PER_COL) * ELEMS_PER_THREAD;

        if (col < TILE_N) {
            const __nv_bfloat16* src = A + (block_row + row_base) + (long long)M * (col_start + col);
            __nv_bfloat16* dst = smem_A + col * TILE_M + row_base;

            // uint4 = 16 bytes = 8 BF16, fully coalesced within each column
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
        }
    }

    __syncthreads();

    // --- WMMA: each warp computes 16x16 Hadamard transform ---
    const int warp_row_start = warp_id * 16;

    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, col_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    // A is col-major in smem: pointer to warp's 16 rows, leading dim = TILE_M
    load_matrix_sync(a_frag, smem_A + warp_row_start, TILE_M);
    load_matrix_sync(b_frag, smem_B, TILE_N);
    fill_fragment(c_frag, 0.0f);
    mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store WMMA result to shared memory (row-major, 16 cols per row)
    float* warp_result = smem_C + warp_id * 16 * 16;
    store_matrix_sync(warp_result, c_frag, 16, mem_row_major);

    // No barrier needed: store_matrix_sync is warp-synchronous,
    // and each warp only reads its own warp_result below.

    // --- Quantize: FP32 → FP4 with per-16-element SFC ---
    const float global_amax_val = *global_amax;
    const float global_encode_scale = compute_global_encode_scale(global_amax_val);
    const float global_decode_scale = 1.0f / global_encode_scale;
    const float scale_multiplier = global_encode_scale / FP4_MAX;

    // Thread mapping: 2 threads per row, 16 rows per warp
    const int row_in_tile = lane_id / 2;
    const int half = lane_id % 2;
    const int global_row = block_row + warp_row_start + row_in_tile;

    if (row_in_tile < 16 && global_row < M) {
        // Load 8 FP32 values from this thread's half of the row
        float vals[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            vals[i] = warp_result[row_in_tile * 16 + half * 8 + i];
        }

        // BF16 round-trip for bitwise compatibility with unfused kernels
        if constexpr (!kUseFastMath) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                vals[i] = __bfloat162float(__float2bfloat16(vals[i]));
            }
        }

        // Compute partial amax over 8 values
        float local_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            local_max = fmaxf(local_max, fabsf(vals[i]));
        }

        // Reduce amax across the 2 threads sharing this row
        float other_max = __shfl_xor_sync(0xFFFFFFFF, local_max, 1);
        float row_max = fmaxf(local_max, other_max);

        // Per-vector scale factor (NVFP4 recipe)
        float pvscale = row_max * scale_multiplier;
        uint8_t pvscale_fp8 = float_to_ue4m3(pvscale);
        float pvscale_dequant = ue4m3_to_float(pvscale_fp8);

        float qpvscale_scaled = pvscale_dequant * global_decode_scale;
        float acc_scale;
        if constexpr (kUseFastMath) {
            acc_scale = __frcp_rn(qpvscale_scaled);
        } else {
            acc_scale = (qpvscale_scaled != 0.0f) ? (1.0f / qpvscale_scaled) : FLT_MAX;
        }
        acc_scale = fminf(acc_scale, FLT_MAX);

        // Scale values
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            vals[i] *= acc_scale;
        }

        // Convert to FP4 and pack.
        // cvt_e2m1x4_rn(a,b,c,d) → bits[3:0]=d, [7:4]=c, [11:8]=b, [15:12]=a
        // Memory packing: byte0 = {col1_hi, col0_lo}, byte1 = {col3_hi, col2_lo}
        // So 16-bit word needs bits[3:0]=col0, [7:4]=col1, [11:8]=col2, [15:12]=col3
        // → call with reversed order: cvt_e2m1x4_rn(col3, col2, col1, col0)
        uint16_t packed_lo, packed_hi;
        if constexpr (kEnableStochasticRounding) {
            const uint64_t rng_seed   = rng_state ? rng_state[0] : 0;
            const uint64_t rng_offset = rng_state ? rng_state[1] : 0;
            const uint64_t rng_seq    = threadIdx.x + (uint64_t)blockIdx.x * blockDim.x
                                      + (uint64_t)blockIdx.y * blockDim.x * gridDim.x;
            Philox4x32 rng;
            rng.init(rng_seed, rng_seq, rng_offset);
            uint4 rand = rng.generate();

            packed_lo = cvt_e2m1x4_sr(vals[3], vals[2], vals[1], vals[0], rand.x);
            packed_hi = cvt_e2m1x4_sr(vals[7], vals[6], vals[5], vals[4], rand.y);
        } else {
            packed_lo = cvt_e2m1x4_rn(vals[3], vals[2], vals[1], vals[0]);
            packed_hi = cvt_e2m1x4_rn(vals[7], vals[6], vals[5], vals[4]);
        }

        uint32_t packed = static_cast<uint32_t>(packed_lo)
                        | (static_cast<uint32_t>(packed_hi) << 16);

        // Write FP4 output: C is row-major, 2 values per byte
        // Byte offset = (global_row * N + col_start + half * 8) / 2
        int byte_offset = (global_row * N + col_start + half * 8) / 2;
        *reinterpret_cast<uint32_t*>(C + byte_offset) = packed;

        // Write SFC: one per row per 16-col group
        if (half == 0) {
            int sfc_idx = global_row * (N / TILE_N) + col_group;
            SFC[sfc_idx] = pvscale_fp8;
        }
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

    dim3 grid(m / TILE_M, n / TILE_N);
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

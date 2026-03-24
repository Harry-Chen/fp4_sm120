#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// ===================================================================
// Type stub — replace with your actual fp4e2m1x4 definition
// ===================================================================
struct fp4e2m1x4 {
    uint16_t data;
};

// ===================================================================
// E2M1 SR noise injection
// ===================================================================
//
// Algorithm: add symmetric noise in [-ULP/2, ULP/2) then clamp to
// x_lo so we stay in the correct ULP bracket. Hardware RN then
// yields P(round_up) = (|x| - |x_lo|) / ULP, i.e. stochastic rounding.

__device__ __forceinline__
float apply_sr_noise_e2m1(float x, unsigned rand_byte) {
    unsigned u = __float_as_uint(x);
    unsigned abs_u = u & 0x7FFFFFFFu;
    unsigned exp = abs_u >> 23;

    // E2M1 ULP: 0.5 for exp<128, 1.0 for exp==128, 2.0 for exp>=129
    unsigned ulp_bexp = min(max(exp, 127u), 129u) - 1u;
    float ulp = __uint_as_float(ulp_bexp << 23);

    // E2M1 truncation of |x| (floor to nearest representable)
    float x_lo_abs;
    if (exp >= 127u) {
        // Normal E2M1: mask off all but 1 mantissa bit
        x_lo_abs = __uint_as_float(abs_u & 0xFFC00000u);
    } else {
        x_lo_abs = (exp >= 126u) ? 0.5f : 0.0f;
    }

    // Symmetric noise in [-ULP/2, ULP/2)
    // (rand_byte - 128) / 256 * ULP
    float noise = (float)((int)rand_byte - 128) * __uint_as_float(0x3B800000u) * ulp;

    // Clamp: prevent noise from pushing below x_lo (ULP boundary crossing)
    float ax_noisy = fmaxf(fabsf(x) + noise, x_lo_abs);

    return copysignf(ax_noisy, x);
}

// ===================================================================
// e2m1x2 wrapper: pack two cvt.rn.satfinite.e2m1x2 into 16 bits
// ===================================================================
//
// Hardware packing of cvt.rn.satfinite.e2m1x2.f32 r8, src1, src2:
//   r8[7:4] = e2m1(src1)    ← high nibble
//   r8[3:0] = e2m1(src2)    ← low nibble
//
// Original e2m1x4 layout: cvt.*.e2m1x4 %0, {a, b, c, d}
//   bits[3:0]=a, [7:4]=b, [11:8]=c, [15:12]=d
//
// To reproduce with two e2m1x2 calls:
//   lo8 = cvt(b, a)  →  lo8[3:0]=a, lo8[7:4]=b   ✓
//   hi8 = cvt(d, c)  →  hi8[3:0]=c, hi8[7:4]=d   ✓
//   result = lo8 | (hi8 << 8)

__device__ __forceinline__
unsigned short cvt_e2m1x4_rn(float a, float b, float c, float d) {
    unsigned short result;
    asm volatile(
        "{\n\t"
        ".reg .b8  lo8, hi8;\n\t"
        ".reg .b32 lo32, hi32, packed32;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 lo8, %2, %1;\n\t"  // (b, a) → a=low, b=high
        "cvt.rn.satfinite.e2m1x2.f32 hi8, %4, %3;\n\t"  // (d, c) → c=low, d=high
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

// ===================================================================
// Main conversion: drop-in replacement
// ===================================================================

__device__ __forceinline__
fp4e2m1x4 mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding(
    const uint64_t in_4x, const float2 scale, const uint32_t rbits) {

    float v0, v1, v2, v3;
    asm volatile(
        "{\n\t"
        ".reg .b16  b0, b1, b2, b3;\n\t"
        ".reg .b64  p01, p23;\n\t"
        "mov.b64 {b0, b1, b2, b3}, %4;\n\t"
        "cvt.f32.bf16 %0, b0;\n\t"
        "cvt.f32.bf16 %1, b1;\n\t"
        "cvt.f32.bf16 %2, b2;\n\t"
        "cvt.f32.bf16 %3, b3;\n\t"
        "mov.b64 p01, {%0, %1};\n\t"
        "mov.b64 p23, {%2, %3};\n\t"
        "mul.f32x2 p01, p01, %5;\n\t"
        "mul.f32x2 p23, p23, %5;\n\t"
        "mov.b64 {%1, %0}, p01;\n\t"
        "mov.b64 {%3, %2}, p23;\n\t"
        "}"
        : "=f"(v0), "=f"(v1), "=f"(v2), "=f"(v3)
        : "l"(in_4x), "l"(reinterpret_cast<const uint64_t &>(scale))
    );

    // Software SR noise (matches original element ordering)
    v2 = apply_sr_noise_e2m1(v2, (rbits      ) & 0xFFu);
    v3 = apply_sr_noise_e2m1(v3, (rbits >>  8) & 0xFFu);
    v0 = apply_sr_noise_e2m1(v0, (rbits >> 16) & 0xFFu);
    v1 = apply_sr_noise_e2m1(v1, (rbits >> 24) & 0xFFu);

    // Original: cvt.rs.satfinite.e2m1x4.f32 %0, {v2, v3, v0, v1}, rbits
    // Layout:   nibble0=v2, nibble1=v3, nibble2=v0, nibble3=v1
    unsigned short out_4x = cvt_e2m1x4_rn(v2, v3, v0, v1);

    fp4e2m1x4 result;
    result.data = out_4x;
    return result;
}

// ===================================================================
// Standalone FP32×4 → E2M1x4 SR (for testing)
// ===================================================================

__device__ __forceinline__
unsigned short fp32x4_to_e2m1x4_sr(float a, float b, float c, float d,
                                     unsigned rbits) {
    a = apply_sr_noise_e2m1(a, (rbits      ) & 0xFFu);
    b = apply_sr_noise_e2m1(b, (rbits >>  8) & 0xFFu);
    c = apply_sr_noise_e2m1(c, (rbits >> 16) & 0xFFu);
    d = apply_sr_noise_e2m1(d, (rbits >> 24) & 0xFFu);
    return cvt_e2m1x4_rn(a, b, c, d);
}

// ===================================================================
// E2M1 decode helper
// ===================================================================

__host__ __device__ __forceinline__
float e2m1_decode(unsigned code4) {
    unsigned sign = (code4 >> 3) & 1u;
    unsigned mag  = code4 & 0x7u;
    const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float v = table[mag];
    return sign ? -v : v;
}

// ===================================================================
// Test 0: Bare cvt instruction sanity check
// ===================================================================

__global__ void test_bare_cvt(unsigned short *out) {
    float vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

    // Use cvt_e2m1x4_rn to test full 4-element pack
    out[0] = cvt_e2m1x4_rn(vals[0], vals[1], vals[2], vals[3]);
    out[1] = cvt_e2m1x4_rn(vals[4], vals[5], vals[6], vals[7]);
}

// ===================================================================
// Test 1: SR probability
// ===================================================================

__global__ void test_sr_probability(
    const float *__restrict__ test_vals,
    unsigned *__restrict__ round_up_count,
    int n_vals,
    int n_trials_per_val
) {
    int vid = blockIdx.x;
    if (vid >= n_vals) return;

    float x = test_vals[vid];
    curandState rng;
    curand_init(42u + vid, threadIdx.x, 0, &rng);

    unsigned local_count = 0;
    int trials_per_thread = n_trials_per_val / blockDim.x;

    for (int i = 0; i < trials_per_thread; i++) {
        unsigned rbits = curand(&rng);
        // Put x in nibble 0 (first arg to fp32x4_to_e2m1x4_sr)
        unsigned short packed = fp32x4_to_e2m1x4_sr(x, 0.0f, 0.0f, 0.0f, rbits);
        unsigned code = packed & 0xFu;  // nibble 0 = first arg
        float decoded = e2m1_decode(code);
        if (fabsf(decoded) > fabsf(x)) {
            local_count++;
        }
    }

    __shared__ unsigned sdata[256];
    sdata[threadIdx.x] = local_count;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&round_up_count[vid], sdata[0]);
    }
}

// ===================================================================
// Test 2: Unbiasedness
// ===================================================================

__global__ void test_sr_unbiasedness(
    const float *__restrict__ test_vals,
    double *__restrict__ sum_out,
    int n_vals,
    int n_trials_per_val
) {
    int vid = blockIdx.x;
    if (vid >= n_vals) return;

    float x = test_vals[vid];
    curandState rng;
    curand_init(123u + vid, threadIdx.x, 0, &rng);

    double local_sum = 0.0;
    int trials_per_thread = n_trials_per_val / blockDim.x;

    for (int i = 0; i < trials_per_thread; i++) {
        unsigned rbits = curand(&rng);
        unsigned short packed = fp32x4_to_e2m1x4_sr(x, 0.0f, 0.0f, 0.0f, rbits);
        unsigned code = packed & 0xFu;
        local_sum += (double)e2m1_decode(code);
    }

    __shared__ double sdata[256];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&sum_out[vid], sdata[0]);
    }
}

// ===================================================================
// Test 3: Exact representable values
// ===================================================================

__global__ void test_exact_values(int *failures) {
    const float exact[] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    curandState rng;
    curand_init(999u, threadIdx.x, 0, &rng);

    int idx = threadIdx.x;
    if (idx >= 16) return;

    float x = exact[idx];
    int local_fail = 0;

    for (int i = 0; i < 1000; i++) {
        unsigned rbits = curand(&rng);
        unsigned short packed = fp32x4_to_e2m1x4_sr(x, 0.0f, 0.0f, 0.0f, rbits);
        unsigned code = packed & 0xFu;
        float decoded = e2m1_decode(code);
        if (decoded != x && !(x == 0.0f && decoded == 0.0f)) {
            local_fail++;
        }
    }

    if (local_fail > 0) {
        atomicAdd(failures, local_fail);
    }
}

// ===================================================================
// Test 4: Saturation
// ===================================================================

__global__ void test_saturation(int *failures) {
    const float big_vals[] = {7.0f, 10.0f, 100.0f, 1e10f, -8.0f, -50.0f};
    int idx = threadIdx.x;
    if (idx >= 6) return;

    curandState rng;
    curand_init(777u, idx, 0, &rng);

    float x = big_vals[idx];
    int local_fail = 0;

    for (int i = 0; i < 1000; i++) {
        unsigned rbits = curand(&rng);
        unsigned short packed = fp32x4_to_e2m1x4_sr(x, 0.0f, 0.0f, 0.0f, rbits);
        unsigned code = packed & 0xFu;
        float decoded = e2m1_decode(code);
        float expected = (x > 0.0f) ? 6.0f : -6.0f;
        if (decoded != expected) {
            local_fail++;
        }
    }

    if (local_fail > 0) {
        atomicAdd(failures, local_fail);
    }
}

// ===================================================================
// Test 5: All 4 lanes valid
// ===================================================================

__global__ void test_all_lanes(int *failures) {
    curandState rng;
    curand_init(555u, threadIdx.x, 0, &rng);

    float vals[] = {0.7f, -1.3f, 2.5f, -4.8f};
    int local_fail = 0;

    for (int i = 0; i < 1000; i++) {
        unsigned rbits = curand(&rng);
        unsigned short packed = fp32x4_to_e2m1x4_sr(
            vals[0], vals[1], vals[2], vals[3], rbits);

        for (int lane = 0; lane < 4; lane++) {
            unsigned code = (packed >> (lane * 4)) & 0xFu;
            float decoded = e2m1_decode(code);
            float mag = fabsf(decoded);
            bool valid = (mag == 0.0f || mag == 0.5f || mag == 1.0f ||
                          mag == 1.5f || mag == 2.0f || mag == 3.0f ||
                          mag == 4.0f || mag == 6.0f);
            bool sign_ok = (decoded == 0.0f) ||
                           ((decoded > 0.0f) == (vals[lane] > 0.0f));
            if (!valid || !sign_ok) local_fail++;
        }
    }

    if (local_fail > 0) atomicAdd(failures, local_fail);
}

// ===================================================================
// Benchmark kernel
// ===================================================================
// Each thread converts one group of 4 floats.
// Uses inline PRNG (no external random buffer → pure compute + read/write).

__device__ __forceinline__
unsigned bench_hash(unsigned seed, unsigned tid) {
    unsigned x = seed ^ (tid * 2654435761u);
    x ^= x >> 16; x *= 0x45d9f3bu;
    x ^= x >> 16; x *= 0x45d9f3bu;
    x ^= x >> 16;
    return x;
}

// Init kernel: fill with varying data to defeat L2 caching of constant patterns
__global__ void bench_init_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // Cheap varying pattern: cast index bits to float-ish range
    unsigned u = (127u << 23) | (idx & 0x7FFFFFu);  // 1.0 .. ~2.0
    float sign = (idx & 0x800000) ? -1.0f : 1.0f;
    data[idx] = __uint_as_float(u) * sign * (float)((idx % 7) + 1);
}

__global__ void bench_fp32_to_e2m1x4_sr(
    const float *__restrict__ input,
    unsigned short *__restrict__ output,
    unsigned seed,
    int n_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_groups) return;

    // Coalesced 128-bit load: adjacent threads read consecutive float4s
    float4 v = reinterpret_cast<const float4 *>(input)[idx];
    unsigned rbits = bench_hash(seed, idx);

    output[idx] = fp32x4_to_e2m1x4_sr(v.x, v.y, v.z, v.w, rbits);
}

// Reduction kernel: compute checksum to prevent dead-code elimination of stores
__global__ void bench_checksum(const unsigned short *data, int n,
                                unsigned long long *out) {
    unsigned long long local_sum = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        local_sum += data[i];
    }
    atomicAdd(out, local_sum);
}

// Read-only bandwidth kernel: same memory footprint as our SR kernel
// but does no compute — just reads FP32 and writes packed bytes.
// This gives the true memory-bound ceiling for our access pattern.
__global__ void bench_readonly_baseline(
    const float *__restrict__ input,
    unsigned short *__restrict__ output,
    int n_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_groups) return;

    float4 v = reinterpret_cast<const float4 *>(input)[idx];
    unsigned a = __float_as_uint(v.x);
    unsigned b = __float_as_uint(v.y);
    unsigned c = __float_as_uint(v.z);
    unsigned d = __float_as_uint(v.w);
    output[idx] = (unsigned short)((a ^ b ^ c ^ d) & 0xFFFFu);
}

// ===================================================================
// Host helpers
// ===================================================================

#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(1);                                                       \
    }                                                                  \
} while(0)

static float e2m1_floor_host(float x) {
    float ax = fabsf(x);
    const float v[] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float lo = 0.0f;
    for (int i = 7; i >= 0; i--) {
        if (ax >= v[i]) { lo = v[i]; break; }
    }
    return copysignf(lo, x);
}

static float e2m1_ceil_host(float x) {
    float ax = fabsf(x);
    const float v[] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    for (int i = 0; i < 8; i++) {
        if (v[i] > ax) return copysignf(v[i], x);
    }
    return copysignf(6.0f, x);
}

// ===================================================================
// main
// ===================================================================

int main() {
    printf("=== E2M1 Stochastic Rounding Tests ===\n\n");

    // ------ Test 0: Bare cvt sanity check ------
    {
        unsigned short *d_out;
        CUDA_CHECK(cudaMalloc(&d_out, 2 * sizeof(unsigned short)));
        test_bare_cvt<<<1, 1>>>(d_out);
        unsigned short h_out[2];
        CUDA_CHECK(cudaMemcpy(h_out, d_out, 2 * sizeof(unsigned short), cudaMemcpyDeviceToHost));

        printf("[Test 0] Bare cvt_e2m1x4_rn sanity check:\n");
        // Expected: cvt_e2m1x4_rn(a,b,c,d) → nibble0=a, nibble1=b, nibble2=c, nibble3=d
        // Pack 0: (0.0, 0.5, 1.0, 1.5) → codes (0, 1, 2, 3) → 0x3210
        // Pack 1: (2.0, 3.0, 4.0, 6.0) → codes (4, 5, 6, 7) → 0x7654
        const unsigned short expect[2] = {0x3210, 0x7654};
        for (int i = 0; i < 2; i++) {
            printf("  pack%d: raw=0x%04x (expect 0x%04x)  ", i, h_out[i], expect[i]);
            for (int n = 0; n < 4; n++) {
                unsigned nib = (h_out[i] >> (n * 4)) & 0xFu;
                unsigned exp_nib = (expect[i] >> (n * 4)) & 0xFu;
                printf("n%d=%u(%u) ", n, nib, exp_nib);
            }
            printf("%s\n", h_out[i] == expect[i] ? "OK" : "MISMATCH");
        }
        printf("\n");
        cudaFree(d_out);
    }

    // ------ Test 3: Exact values ------
    {
        int *d_fail;
        CUDA_CHECK(cudaMalloc(&d_fail, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_fail, 0, sizeof(int)));
        test_exact_values<<<1, 16>>>(d_fail);
        int h_fail = 0;
        CUDA_CHECK(cudaMemcpy(&h_fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost));
        printf("[Test 3] Exact representable values: %s (%d failures / 16000 trials)\n",
               h_fail == 0 ? "PASS" : "FAIL", h_fail);
        cudaFree(d_fail);
    }

    // ------ Test 4: Saturation ------
    {
        int *d_fail;
        CUDA_CHECK(cudaMalloc(&d_fail, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_fail, 0, sizeof(int)));
        test_saturation<<<1, 6>>>(d_fail);
        int h_fail = 0;
        CUDA_CHECK(cudaMemcpy(&h_fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost));
        printf("[Test 4] Saturation (|x| > 6.0):    %s (%d failures / 6000 trials)\n",
               h_fail == 0 ? "PASS" : "FAIL", h_fail);
        cudaFree(d_fail);
    }

    // ------ Test 5: All lanes valid ------
    {
        int *d_fail;
        CUDA_CHECK(cudaMalloc(&d_fail, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_fail, 0, sizeof(int)));
        test_all_lanes<<<1, 1>>>(d_fail);
        int h_fail = 0;
        CUDA_CHECK(cudaMemcpy(&h_fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost));
        printf("[Test 5] All 4 lanes valid output:   %s (%d failures / 4000 trials)\n",
               h_fail == 0 ? "PASS" : "FAIL", h_fail);
        cudaFree(d_fail);
    }

    // ------ Test 1: SR probability ------
    {
        const float h_vals[] = {
            0.1f, 0.25f, 0.3f, 0.75f,
            1.2f, 1.7f,
            2.3f, 2.7f, 3.5f,
            4.5f, 5.0f, 5.9f,
            -0.3f, -1.7f, -3.5f, -5.0f
        };
        int n_vals = sizeof(h_vals) / sizeof(h_vals[0]);
        int n_trials = 256 * 10000;
        int block = 256;

        float *d_vals;
        unsigned *d_counts;
        CUDA_CHECK(cudaMalloc(&d_vals, n_vals * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_counts, n_vals * sizeof(unsigned)));
        CUDA_CHECK(cudaMemcpy(d_vals, h_vals, n_vals * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_counts, 0, n_vals * sizeof(unsigned)));

        test_sr_probability<<<n_vals, block>>>(d_vals, d_counts, n_vals, n_trials);

        unsigned h_counts[32];
        CUDA_CHECK(cudaMemcpy(h_counts, d_counts, n_vals * sizeof(unsigned), cudaMemcpyDeviceToHost));

        printf("\n[Test 1] SR probability (expected vs measured):\n");
        printf("  %-8s  %-8s  %-8s  %-8s  %-10s  %-10s  %s\n",
               "x", "x_lo", "x_hi", "ULP", "P_expect", "P_measured", "Status");
        int prob_pass = 0;
        for (int i = 0; i < n_vals; i++) {
            float x = h_vals[i];
            float lo = e2m1_floor_host(x);
            float hi = e2m1_ceil_host(x);
            float ulp = fabsf(hi) - fabsf(lo);
            float p_expect = (ulp > 0.0f) ? (fabsf(x) - fabsf(lo)) / ulp : 0.0f;
            float p_meas = (float)h_counts[i] / (float)n_trials;
            float err = fabsf(p_meas - p_expect);
            bool ok = err < 0.015f;
            printf("  %+7.3f  %+7.3f  %+7.3f  %5.2f    %9.5f   %9.5f   %s (err=%.4f)\n",
                   x, lo, hi, ulp, p_expect, p_meas, ok ? "OK" : "FAIL", err);
            prob_pass += ok;
        }
        printf("  %d / %d passed\n", prob_pass, n_vals);
        cudaFree(d_vals);
        cudaFree(d_counts);
    }

    // ------ Test 2: Unbiasedness ------
    {
        const float h_vals[] = {
            0.1f, 0.3f, 0.75f, 1.2f, 1.7f,
            2.3f, 3.5f, 5.0f,
            -0.3f, -1.7f, -3.5f, -5.0f
        };
        int n_vals = sizeof(h_vals) / sizeof(h_vals[0]);
        int n_trials = 256 * 10000;
        int block = 256;

        float *d_vals;
        double *d_sums;
        CUDA_CHECK(cudaMalloc(&d_vals, n_vals * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sums, n_vals * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_vals, h_vals, n_vals * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_sums, 0, n_vals * sizeof(double)));

        test_sr_unbiasedness<<<n_vals, block>>>(d_vals, d_sums, n_vals, n_trials);

        double h_sums[32];
        CUDA_CHECK(cudaMemcpy(h_sums, d_sums, n_vals * sizeof(double), cudaMemcpyDeviceToHost));

        printf("\n[Test 2] Unbiasedness E[SR(x)] ≈ x:\n");
        printf("  %-8s  %-12s  %-12s  %s\n", "x", "E[SR(x)]", "error", "Status");
        int unbias_pass = 0;
        for (int i = 0; i < n_vals; i++) {
            float x = h_vals[i];
            double mean = h_sums[i] / (double)n_trials;
            double err = fabs(mean - (double)x);
            bool ok = err < 0.02;
            printf("  %+7.3f  %+11.6f  %10.6f   %s\n",
                   x, mean, err, ok ? "OK" : "FAIL");
            unbias_pass += ok;
        }
        printf("  %d / %d passed\n", unbias_pass, n_vals);
        cudaFree(d_vals);
        cudaFree(d_sums);
    }

    // ------ Test 6: Throughput benchmark ------
    {
        printf("\n[Test 6] Throughput benchmark:\n");

        const size_t sizes[] = {
            1u << 20,    //  1M floats =   4 MB
            16u << 20,   // 16M floats =  64 MB
            64u << 20,   // 64M floats = 256 MB
        };
        const int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
        const int warmup = 20;
        const int iters  = 100;
        const int block  = 256;

        for (int si = 0; si < n_sizes; si++) {
            size_t n_floats = sizes[si];
            size_t n_groups = n_floats / 4;

            float *d_in;
            unsigned short *d_out;
            unsigned long long *d_cksum;
            CUDA_CHECK(cudaMalloc(&d_in,  n_floats * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_out, n_groups * sizeof(unsigned short)));
            CUDA_CHECK(cudaMalloc(&d_cksum, sizeof(unsigned long long)));

            // Fill with varying data to defeat L2 constant-pattern caching
            {
                int ig = (int)((n_floats + 255) / 256);
                bench_init_kernel<<<ig, 256>>>(d_in, (int)n_floats);
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            int grid = (int)((n_groups + block - 1) / block);

            // --- Baseline: read 4 floats + write 2 bytes, no compute ---
            cudaEvent_t bs_start, bs_stop;
            CUDA_CHECK(cudaEventCreate(&bs_start));
            CUDA_CHECK(cudaEventCreate(&bs_stop));

            for (int i = 0; i < warmup; i++)
                bench_readonly_baseline<<<grid, block>>>(d_in, d_out, (int)n_groups);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaEventRecord(bs_start));
            for (int i = 0; i < iters; i++)
                bench_readonly_baseline<<<grid, block>>>(d_in, d_out, (int)n_groups);
            CUDA_CHECK(cudaEventRecord(bs_stop));
            CUDA_CHECK(cudaEventSynchronize(bs_stop));

            float bs_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&bs_ms, bs_start, bs_stop));
            float bs_ms_per = bs_ms / (float)iters;

            // Force output to be consumed
            CUDA_CHECK(cudaMemset(d_cksum, 0, sizeof(unsigned long long)));
            bench_checksum<<<64, 256>>>(d_out, (int)n_groups, d_cksum);

            // --- SR kernel ---
            cudaEvent_t sr_start, sr_stop;
            CUDA_CHECK(cudaEventCreate(&sr_start));
            CUDA_CHECK(cudaEventCreate(&sr_stop));

            for (int i = 0; i < warmup; i++)
                bench_fp32_to_e2m1x4_sr<<<grid, block>>>(
                    d_in, d_out, 42u + i, (int)n_groups);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaEventRecord(sr_start));
            for (int i = 0; i < iters; i++)
                bench_fp32_to_e2m1x4_sr<<<grid, block>>>(
                    d_in, d_out, 12345u + i, (int)n_groups);
            CUDA_CHECK(cudaEventRecord(sr_stop));
            CUDA_CHECK(cudaEventSynchronize(sr_stop));

            float sr_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&sr_ms, sr_start, sr_stop));
            float sr_ms_per = sr_ms / (float)iters;

            // Force output to be consumed
            CUDA_CHECK(cudaMemset(d_cksum, 0, sizeof(unsigned long long)));
            bench_checksum<<<64, 256>>>(d_out, (int)n_groups, d_cksum);
            unsigned long long h_cksum;
            CUDA_CHECK(cudaMemcpy(&h_cksum, d_cksum, sizeof(h_cksum), cudaMemcpyDeviceToHost));

            // Bytes: read n_floats*4, write n_groups*2
            double total_bytes = (double)n_floats * 4.0 + (double)n_groups * 2.0;
            double bs_gbps = total_bytes / (bs_ms_per * 1e6);
            double sr_gbps = total_bytes / (sr_ms_per * 1e6);
            double efficiency = sr_gbps / bs_gbps * 100.0;

            printf("  %6.1f MB FP32 in | baseline %7.3f ms %7.1f GB/s | "
                   "SR kernel %7.3f ms %7.1f GB/s | eff %.0f%% (cksum=0x%llx)\n",
                   (double)n_floats * 4.0 / (1024.0 * 1024.0),
                   bs_ms_per, bs_gbps,
                   sr_ms_per, sr_gbps,
                   efficiency, h_cksum);

            CUDA_CHECK(cudaEventDestroy(bs_start));
            CUDA_CHECK(cudaEventDestroy(bs_stop));
            CUDA_CHECK(cudaEventDestroy(sr_start));
            CUDA_CHECK(cudaEventDestroy(sr_stop));
            cudaFree(d_in);
            cudaFree(d_out);
            cudaFree(d_cksum);
        }
    }

    printf("\n=== Done ===\n");
    return 0;
}

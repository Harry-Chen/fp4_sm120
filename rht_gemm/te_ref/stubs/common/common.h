// Minimal stubs for TE common.h — just enough for rht_gemm_ntt_w_sfc.
// Replaces the real TE header which pulls in cuDNN and many other deps.
#pragma once
#include <cstdio>
#include <cassert>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define FP4_TYPE_SUPPORTED (CUDA_VERSION >= 12080)
#if FP4_TYPE_SUPPORTED
#include <cuda_fp4.h>
#endif

#define NVTE_CHECK(cond, ...)                                    \
    do {                                                          \
        if (!(cond)) {                                            \
            fprintf(stderr, "NVTE_CHECK failed: %s\n", #cond);   \
            abort();                                              \
        }                                                         \
    } while (0)

#define NVTE_CHECK_CUDA(call)                                    \
    do {                                                          \
        auto _err = (call);                                       \
        if (_err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",         \
                    cudaGetErrorString(_err), __FILE__, __LINE__);\
            abort();                                              \
        }                                                         \
    } while (0)

#define NVTE_DEVICE_ERROR(...) printf(__VA_ARGS__)
#define NVTE_API_CALL(name) (void)0

// ARCH_BLACKWELL_FAMILY and ARCH_HAS_STOCHASTIC_ROUNDING are defined
// in common/util/ptx.cuh (the real TE header, symlinked).

// Type aliases used by ptx.cuh and the reference kernel
namespace transformer_engine {
using byte = uint8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;
using fp32 = float;
using fp16 = half;
using bf16 = nv_bfloat16;
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;
#if CUDA_VERSION >= 12080
using fp8e8m0 = __nv_fp8_e8m0;
#endif
using e8m0_t = uint8_t;
#if FP4_TYPE_SUPPORTED
using fp4e2m1 = __nv_fp4_e2m1;
using fp4e2m1x2 = __nv_fp4x2_e2m1;
using fp4e2m1x4 = __nv_fp4x4_e2m1;
#endif
}  // namespace transformer_engine

#ifndef NVTE_BUILD_NUM_PHILOX_ROUNDS
#define NVTE_BUILD_NUM_PHILOX_ROUNDS 10
#endif

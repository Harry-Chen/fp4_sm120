// Separate TU for our SM120 kernel — avoids include conflicts with TE headers.
#include "rht_gemm_sm120.cuh"
#include <cuda_bf16.h>
#include <cstdint>

// <SR=false, FastMath=false>
void run_ours(int m, int n,
              const __nv_bfloat16* A, const __nv_bfloat16* B,
              uint8_t* C, uint8_t* SFC,
              const float* global_amax, const size_t* rng_state,
              uint32_t sm_count, cudaStream_t stream) {
    rht_gemm_sm120::rht_gemm_ntt_w_sfc<
        __nv_bfloat16, __nv_bfloat16, uint8_t, uint8_t, false, false>(
        m, n, A, B, C, SFC, global_amax, rng_state, sm_count, stream);
}

// <SR=false, FastMath=true>
void run_ours_fast(int m, int n,
                   const __nv_bfloat16* A, const __nv_bfloat16* B,
                   uint8_t* C, uint8_t* SFC,
                   const float* global_amax, const size_t* rng_state,
                   uint32_t sm_count, cudaStream_t stream) {
    rht_gemm_sm120::rht_gemm_ntt_w_sfc<
        __nv_bfloat16, __nv_bfloat16, uint8_t, uint8_t, false, true>(
        m, n, A, B, C, SFC, global_amax, rng_state, sm_count, stream);
}

// <SR=true, FastMath=false>
void run_ours_sr(int m, int n,
                 const __nv_bfloat16* A, const __nv_bfloat16* B,
                 uint8_t* C, uint8_t* SFC,
                 const float* global_amax, const size_t* rng_state,
                 uint32_t sm_count, cudaStream_t stream) {
    rht_gemm_sm120::rht_gemm_ntt_w_sfc<
        __nv_bfloat16, __nv_bfloat16, uint8_t, uint8_t, true, false>(
        m, n, A, B, C, SFC, global_amax, rng_state, sm_count, stream);
}

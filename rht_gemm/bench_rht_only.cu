// Standalone RHT GEMM benchmark for ncu/nsys profiling.
// Runs only rht_gemm_ntt_w_sfc at a configurable size, many iterations.
#include "rht_gemm_sm120.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(1);                                                       \
    }                                                                  \
} while(0)

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 8192;
    int N = (argc > 2) ? atoi(argv[2]) : 10240;
    int iters = (argc > 3) ? atoi(argv[3]) : 50;

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<__nv_bfloat16> h_A(M * N);
    std::vector<__nv_bfloat16> h_B(16 * 16);
    for (int i = 0; i < M * N; i++) h_A[i] = __float2bfloat16(dist(gen));
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++) {
            int sign = __builtin_popcount(r & c) % 2 == 0 ? 1 : -1;
            h_B[r * 16 + c] = __float2bfloat16(sign * 0.25f);
        }
    float global_amax_val = 4.0f;

    __nv_bfloat16 *d_A, *d_B;
    uint8_t *d_C, *d_SFC;
    float *d_global_amax;
    size_t *d_rng_state;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)M * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_B, 256 * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_C, (size_t)M * N / 2));
    CUDA_CHECK(cudaMalloc(&d_SFC, (size_t)M * (N / 16)));
    CUDA_CHECK(cudaMalloc(&d_global_amax, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rng_state, 2 * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), (size_t)M * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), 256 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_global_amax, &global_amax_val, sizeof(float), cudaMemcpyHostToDevice));
    size_t rng_state[2] = {12345, 0};
    CUDA_CHECK(cudaMemcpy(d_rng_state, rng_state, 2 * sizeof(size_t), cudaMemcpyHostToDevice));

    uint32_t sm_count = 170;

    // warmup
    for (int i = 0; i < 5; i++)
        rht_gemm_sm120::rht_gemm_ntt_w_sfc<__nv_bfloat16, __nv_bfloat16, uint8_t, uint8_t, false, false>(
            M, N, d_A, d_B, d_C, d_SFC, d_global_amax, d_rng_state, sm_count, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        rht_gemm_sm120::rht_gemm_ntt_w_sfc<__nv_bfloat16, __nv_bfloat16, uint8_t, uint8_t, false, false>(
            M, N, d_A, d_B, d_C, d_SFC, d_global_amax, d_rng_state, sm_count, 0);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float ms_per = ms / iters;

    // bytes: read A (M*N*2) + B (256*2) + amax(4) + rng(16); write C (M*N/2) + SFC (M*N/16)
    double bytes = (double)M * N * 2.0 + 256*2 + 4 + 16 + (double)M * N / 2.0 + (double)M * (N / 16);
    double gbps = bytes / (ms_per * 1e6);
    double gflops = 2.0 * M * N * 16 / (ms_per * 1e6);
    printf("RHT-only: M=%d N=%d iters=%d | %.4f ms | %.1f GB/s | %.1f GFLOPS\n",
           M, N, iters, ms_per, gbps, gflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_SFC);
    cudaFree(d_global_amax); cudaFree(d_rng_state);
    return 0;
}

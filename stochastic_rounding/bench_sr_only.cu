// Standalone SR kernel benchmark for ncu/nsys profiling.
// Runs only bench_fp32_to_e2m1x4_sr at a large size, many iterations.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#include "sr.sm120.cuh"

#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(1);                                                       \
    }                                                                  \
} while(0)

__global__ void bench_init_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned u = (127u << 23) | (idx & 0x7FFFFFu);
    float sign = (idx & 0x800000) ? -1.0f : 1.0f;
    data[idx] = __uint_as_float(u) * sign * (float)((idx % 7) + 1);
}

__device__ __forceinline__ unsigned bench_hash(unsigned seed, unsigned idx) {
    unsigned h = seed ^ idx;
    h ^= h >> 16; h *= 0x85ebca6bu;
    h ^= h >> 13; h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

__global__ void bench_fp32_to_e2m1x4_sr(
    const float *__restrict__ input,
    unsigned short *__restrict__ output,
    unsigned seed,
    int n_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_groups) return;
    float4 v = reinterpret_cast<const float4 *>(input)[idx];
    unsigned rbits = bench_hash(seed, idx);
    output[idx] = fp32x4_to_e2m1x4_sr(v.x, v.y, v.z, v.w, rbits);
}

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

int main(int argc, char **argv) {
    size_t n_floats = (argc > 1) ? (size_t)atoll(argv[1]) : (64u << 20);
    int iters = (argc > 2) ? atoi(argv[2]) : 200;
    int block = (argc > 3) ? atoi(argv[3]) : 256;

    size_t n_groups = n_floats / 4;
    int grid = (int)((n_groups + block - 1) / block);

    float *d_in;
    unsigned short *d_out;
    unsigned long long *d_cksum;
    CUDA_CHECK(cudaMalloc(&d_in,  n_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n_groups * sizeof(unsigned short)));
    CUDA_CHECK(cudaMalloc(&d_cksum, sizeof(unsigned long long)));

    {
        int ig = (int)((n_floats + 255) / 256);
        bench_init_kernel<<<ig, 256>>>(d_in, (int)n_floats);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int i = 0; i < 10; i++)
        bench_fp32_to_e2m1x4_sr<<<grid, block>>>(d_in, d_out, 42u + i, (int)n_groups);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        bench_fp32_to_e2m1x4_sr<<<grid, block>>>(d_in, d_out, 12345u + i, (int)n_groups);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float ms_per = ms / iters;

    CUDA_CHECK(cudaMemset(d_cksum, 0, sizeof(unsigned long long)));
    bench_checksum<<<64, 256>>>(d_out, (int)n_groups, d_cksum);
    unsigned long long cksum;
    CUDA_CHECK(cudaMemcpy(&cksum, d_cksum, sizeof(cksum), cudaMemcpyDeviceToHost));

    double total_bytes = (double)n_floats * 4.0 + (double)n_groups * 2.0;
    double gbps = total_bytes / (ms_per * 1e6);
    printf("SR-only: n_floats=%zu block=%d iters=%d | %.4f ms | %.1f GB/s | cksum=0x%llx\n",
           n_floats, block, iters, ms_per, gbps, cksum);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_cksum);
    return 0;
}

// Comparison test: SM120 kernel vs SM100 reference (rht_gemm_ntt_w_sfc).
//
// This test is designed to run on a machine with BOTH SM100 and SM120 GPUs
// (e.g., GB200 + RTX 5090), or compiled twice for each architecture.
//
// Build for SM100 (for GB200): make test_compare.exe
// Build for SM120 (for 5090):  make test_compare.exe CUDA_ARCH=120a
//
// When compiled for SM120, this test only runs the SM120 kernel and
// saves the output to a file. The SM100 version does the same. The
// outputs can then be compared offline.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <fstream>

#include "rht_gemm_sm120.cuh"

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,         \
                    __LINE__, cudaGetErrorString(err));                      \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

void run_and_save(int M, int N, const char* output_prefix) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<__nv_bfloat16> h_A(M * N);
    std::vector<__nv_bfloat16> h_B(16 * 16);

    for (int i = 0; i < M * N; i++)
        h_A[i] = __float2bfloat16(dist(gen));

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

    CHECK_CUDA(cudaMalloc(&d_A, M * N * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_B, 256 * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N / 2));
    CHECK_CUDA(cudaMalloc(&d_SFC, M * (N / 16)));
    CHECK_CUDA(cudaMalloc(&d_global_amax, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_rng_state, 2 * sizeof(size_t)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), 256 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_global_amax, &global_amax_val, sizeof(float), cudaMemcpyHostToDevice));

    size_t rng_state[2] = {12345, 0};
    CHECK_CUDA(cudaMemcpy(d_rng_state, rng_state, 2 * sizeof(size_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, M * N / 2));
    CHECK_CUDA(cudaMemset(d_SFC, 0, M * (N / 16)));

    // Run SM120 kernel
    rht_gemm_sm120::rht_gemm_ntt_w_sfc<__nv_bfloat16, __nv_bfloat16, uint8_t, uint8_t, false, false>(
        M, N, d_A, d_B, d_C, d_SFC, d_global_amax, d_rng_state, 170, 0);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back
    std::vector<uint8_t> h_C(M * N / 2);
    std::vector<uint8_t> h_SFC(M * (N / 16));
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N / 2, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_SFC.data(), d_SFC, M * (N / 16), cudaMemcpyDeviceToHost));

    // Save to files
    char fname[256];
    snprintf(fname, sizeof(fname), "%s_C_%dx%d.bin", output_prefix, M, N);
    {
        std::ofstream f(fname, std::ios::binary);
        f.write(reinterpret_cast<char*>(h_C.data()), h_C.size());
        printf("Saved FP4 output to %s (%zu bytes)\n", fname, h_C.size());
    }
    snprintf(fname, sizeof(fname), "%s_SFC_%dx%d.bin", output_prefix, M, N);
    {
        std::ofstream f(fname, std::ios::binary);
        f.write(reinterpret_cast<char*>(h_SFC.data()), h_SFC.size());
        printf("Saved SFC to %s (%zu bytes)\n", fname, h_SFC.size());
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_SFC));
    CHECK_CUDA(cudaFree(d_global_amax));
    CHECK_CUDA(cudaFree(d_rng_state));
}

void compare_files(const char* file1, const char* file2, const char* label) {
    std::ifstream f1(file1, std::ios::binary);
    std::ifstream f2(file2, std::ios::binary);

    if (!f1.is_open()) { printf("Cannot open %s\n", file1); return; }
    if (!f2.is_open()) { printf("Cannot open %s\n", file2); return; }

    std::vector<uint8_t> d1((std::istreambuf_iterator<char>(f1)), std::istreambuf_iterator<char>());
    std::vector<uint8_t> d2((std::istreambuf_iterator<char>(f2)), std::istreambuf_iterator<char>());

    if (d1.size() != d2.size()) {
        printf("%s: size mismatch (%zu vs %zu)\n", label, d1.size(), d2.size());
        return;
    }

    int mismatches = 0;
    for (size_t i = 0; i < d1.size(); i++) {
        if (d1[i] != d2[i]) mismatches++;
    }

    printf("%s: %d / %zu mismatches (%.4f%%)\n",
           label, mismatches, d1.size(), 100.0 * mismatches / d1.size());
}

int main(int argc, char** argv) {
    printf("=== RHT GEMM SM120 vs SM100 Comparison ===\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    if (argc >= 2 && strcmp(argv[1], "--compare") == 0) {
        // Compare mode: compare two sets of output files
        if (argc < 5) {
            printf("Usage: %s --compare <prefix1> <prefix2> <MxN>\n", argv[0]);
            return 1;
        }
        const char* prefix1 = argv[2];
        const char* prefix2 = argv[3];
        int M, N;
        sscanf(argv[4], "%dx%d", &M, &N);

        char f1[256], f2[256];
        snprintf(f1, 256, "%s_C_%dx%d.bin", prefix1, M, N);
        snprintf(f2, 256, "%s_C_%dx%d.bin", prefix2, M, N);
        compare_files(f1, f2, "FP4 output");

        snprintf(f1, 256, "%s_SFC_%dx%d.bin", prefix1, M, N);
        snprintf(f2, 256, "%s_SFC_%dx%d.bin", prefix2, M, N);
        compare_files(f1, f2, "SFC output");
        return 0;
    }

    // Generate mode: run the kernel and save outputs
    const char* prefix = "sm120";
    if (argc >= 2) prefix = argv[1];

    struct TestCase { int m, n; };
    TestCase tests[] = {
        {256, 128},
        {1024, 1024},
        {8192, 5120},
    };

    for (auto& tc : tests) {
        printf("--- %d x %d ---\n", tc.m, tc.n);
        run_and_save(tc.m, tc.n, prefix);
        printf("\n");
    }

    printf("Usage: Run on SM100 with prefix 'sm100', then compare:\n");
    printf("  ./test_compare.exe sm120    # on RTX 5090\n");
    printf("  ./test_compare.exe sm100    # on GB200\n");
    printf("  ./test_compare.exe --compare sm100 sm120 8192x5120\n");

    return 0;
}

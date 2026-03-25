// Minimal stub for TE common/util/cuda_runtime.h
#pragma once
#include <cuda_runtime.h>

namespace transformer_engine {
namespace cuda {

inline int sm_count(int device_id = -1) {
    if (device_id < 0) cudaGetDevice(&device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    return prop.multiProcessorCount;
}

}  // namespace cuda

inline void checkCuDriverContext(cudaStream_t) {}

}  // namespace transformer_engine

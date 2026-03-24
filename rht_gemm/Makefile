CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= 120a
CUTLASS_HOME ?= /home/harry/cutlass

NVCC := $(CUDA_HOME)/bin/nvcc
NVCC_CODE_FLAGS := -gencode=arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
NVCC_COMMON_FLAGS := -Xptxas=-v -O3 -std=c++20 -g -I$(CUTLASS_HOME)/include
NVCC_FLAGS := $(NVCC_CODE_FLAGS) $(NVCC_COMMON_FLAGS)

all: test_correctness.exe

test_correctness.exe: test_correctness.cu rht_gemm_sm120.cuh
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# For SM100 comparison test (requires B200/GB200 hardware + CUTLASS)
test_compare.exe: test_compare.cu rht_gemm_sm120.cuh
	$(NVCC) -gencode=arch=compute_100a,code=sm_100a $(NVCC_COMMON_FLAGS) $< -o $@

%.ptx: %.cu rht_gemm_sm120.cuh
	$(NVCC) $(NVCC_FLAGS) -ptx $< -o $@

.PHONY: all clean

clean:
	rm -rf *.exe *.ptx

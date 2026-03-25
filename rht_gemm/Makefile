CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= 120a
CUTLASS_HOME ?= /home/harry/cutlass

NVCC := $(CUDA_HOME)/bin/nvcc
NVCC_CODE_FLAGS := -gencode=arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
NVCC_COMMON_FLAGS := -Xptxas=-v -O3 -std=c++20 -g -I$(CUTLASS_HOME)/include -I$(CUTLASS_HOME)/tools/util/include
NVCC_FLAGS := $(NVCC_CODE_FLAGS) $(NVCC_COMMON_FLAGS)

all: test_correctness.exe

test_correctness.exe: test_correctness.cu rht_gemm_sm120.cuh
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# SM100 comparison test: our kernel vs TE reference, run on GB200.
# Compiled as 3 TUs to avoid include conflicts between sr.sm120.cuh and TE ptx.cuh.
TE_HOME ?= /home/harry/TransformerEngine
TE_INC := -Ite_ref/stubs -I$(TE_HOME)/transformer_engine -I$(TE_HOME)/transformer_engine/common/include
SM100_FLAGS := -gencode=arch=compute_100a,code=sm_100a $(NVCC_COMMON_FLAGS) -rdc=true --expt-relaxed-constexpr

test_compare.exe: test_compare.cu test_compare_ours.cu test_compare_ref.cu rht_gemm_sm120.cuh
	$(NVCC) $(SM100_FLAGS) -c test_compare.cu -o test_compare_main.o
	$(NVCC) $(SM100_FLAGS) -c test_compare_ours.cu -o test_compare_ours.o
	$(NVCC) $(SM100_FLAGS) $(TE_INC) -c test_compare_ref.cu -o test_compare_ref.o
	$(NVCC) $(SM100_FLAGS) test_compare_main.o test_compare_ours.o test_compare_ref.o -o $@
	rm -f test_compare_main.o test_compare_ours.o test_compare_ref.o

%.ptx: %.cu rht_gemm_sm120.cuh
	$(NVCC) $(NVCC_FLAGS) -ptx $< -o $@

.PHONY: all clean

clean:
	rm -rf *.exe *.ptx

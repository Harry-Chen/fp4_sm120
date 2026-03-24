CUDA_HOME ?= /usr/local/cuda
NVCC := $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS := -gencode=arch=compute_120a,code=sm_120a -Xptxas=-v -O3 -std=c++20 -g

SRC := $(wildcard *.cu)
EXE := $(SRC:%.cu=%.exe)

all: $(EXE)

%.exe: %.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

.PHONY: all clean

clean:
	rm -rf *.exe


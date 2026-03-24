# Random Hadamard Transform GEMM for SM120

SM120 port of the Random Hadamard Transform (RHT) GEMM kernel used by
Transformer Engine's FP4 training recipe.

## Status

Planned — not yet implemented.

## Background

The FP4 training recipe in Transformer Engine applies a random Hadamard
rotation to activations and weights before FP4 quantization. This improves
quantization quality by spreading outlier values across dimensions. The
existing SM100 kernel needs to be ported to SM120 family GPUs.

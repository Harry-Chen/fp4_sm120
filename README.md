# NVFP4 Support for SM120 Family

Patches and kernels to enable Transformer Engine's NVFP4 (FP4 E2M1) training
on NVIDIA SM120 family GPUs (RTX 50x0, DGX Spark), where several SM100-era
PTX instructions are missing or changed.

## Background

Transformer Engine uses FP4 E2M1 quantization for forward-pass activations and
weights in FP4 training recipes. The kernels rely on SM100-specific PTX
instructions that are not all available on the SM120 family:

| Instruction | SM100 | SM120 | Status |
|-------------|-------|-------|--------|
| `cvt.rs.satfinite.e2m1x4.f32` | Yes | **No** | Polyfilled in `stochastic_rounding/` |
| `cvt.rn.satfinite.e2m1x2.f32` | Yes | Yes | Used by polyfill |
| RHT GEMM (Random Hadamard Transform) | SM100 kernel | Needs porting | Planned in `rht_gemm/` |

## Components

### [`stochastic_rounding/`](stochastic_rounding/) — Done

Drop-in polyfills for the stochastic rounding FP4 conversion functions removed
in SM120. Two implementations:

- **`sr.sm120.cuh`** — Uses `cvt.rn.satfinite.e2m1x2.f32` + software SR noise
  injection. Requires SM120 family hardware.
- **`sr.software.cuh`** — Pure software quantization. Works on any CUDA
  architecture.

Polyfilled functions:
- `mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding`
- `mul_cvt_fp32_to_fp4_4x_with_stochastic_rounding`
- `cvt_fp32_to_fp4_4x_with_stochastic_rounding`
- `mul_cvt_bf16_to_fp4_8x_stochastic_rounding`

Validated against native SM100 hardware (`compare.cu` on B300) — bit-exact
match for the SM120 polyfill, statistical equivalence for the software polyfill.

See [`stochastic_rounding/README.md`](stochastic_rounding/README.md) for details.

### [`rht_gemm/`](rht_gemm/) — Planned

Random Hadamard Transform GEMM kernel for SM120. Required by Transformer
Engine's FP4 recipe to apply the Hadamard rotation before quantization.

## Building

Each component has its own Makefile. See the README in each subdirectory.

```bash
# Build stochastic rounding tests (SM120)
cd stochastic_rounding && make

# Build comparison test (requires SM100 hardware)
cd stochastic_rounding && make compare.exe CUDA_ARCH=100a
```

## License

Apache 2.0

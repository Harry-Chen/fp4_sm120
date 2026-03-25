// Separate TU for the TE SM100 reference kernel.
// Compiled with TE stub headers (te_stubs/) that replace the heavy
// TE common.h dependency chain.

// The te_ref file ends with hadamard_transform_cast_fusion_columnwise()
// and nvte_*() C API wrappers that need full TE Tensor types. We stop
// the include before those by defining a guard macro.
#define TE_REF_SKIP_TENSOR_API

// ---- Standard CUDA + CUTLASS includes (same as te_ref needs) ----
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include <cuda/barrier>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>

// TE stubs (resolved via -Ite_stubs before TE paths)
#include "common/common.h"
#include "common/util/cuda_runtime.h"
#include "common/util/curanddx.hpp"
#include "common/util/ptx.cuh"
#include "common/utils.cuh"

#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/builders/sm100_common.inl"
#include "cutlass/numeric_conversion.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/print_error.hpp"

// clang-format off

// ======== Begin extracted TE core (lines 34-723 of the reference) ========
namespace transformer_engine {
namespace detail {
namespace {

using namespace cute;
using cute::Tensor;

#include "te_ref/hadamard_transform_cast_fusion_core.inc"

}  // namespace
}  // namespace detail
}  // namespace transformer_engine

// clang-format on

// ---- Concrete wrappers callable from the main TU ----
using TA   = cute::bfloat16_t;
using TB   = cute::bfloat16_t;
using TC   = cutlass::float_e2m1_t;
using TSFC = cutlass::float_ue4m3_t;

// Call rht_gemm_ntt_w_sfc directly (NOT ttt_wrapper which swaps m/n).
#define LAUNCH_REF(SR, FM)                                                     \
    transformer_engine::detail::rht_gemm_ntt_w_sfc<TA, TB, TC, TSFC, SR, FM>( \
        m, n, reinterpret_cast<TA const*>(A), reinterpret_cast<TB const*>(B),  \
        reinterpret_cast<TC*>(C), reinterpret_cast<TSFC*>(SFC),               \
        global_amax, rng_state, sm_count, stream)

// <SR=false, FastMath=false>
void run_ref(int m, int n,
             const __nv_bfloat16* A, const __nv_bfloat16* B,
             uint8_t* C, uint8_t* SFC,
             const float* global_amax, const size_t* rng_state,
             uint32_t sm_count, cudaStream_t stream) { LAUNCH_REF(false, false); }

// <SR=false, FastMath=true>
void run_ref_fast(int m, int n,
                  const __nv_bfloat16* A, const __nv_bfloat16* B,
                  uint8_t* C, uint8_t* SFC,
                  const float* global_amax, const size_t* rng_state,
                  uint32_t sm_count, cudaStream_t stream) { LAUNCH_REF(false, true); }

// <SR=true, FastMath=false>
void run_ref_sr(int m, int n,
                const __nv_bfloat16* A, const __nv_bfloat16* B,
                uint8_t* C, uint8_t* SFC,
                const float* global_amax, const size_t* rng_state,
                uint32_t sm_count, cudaStream_t stream) { LAUNCH_REF(true, false); }

#undef LAUNCH_REF

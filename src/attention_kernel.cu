#pragma nv_diag_suppress 174
#pragma nv_diag_suppress 307
#pragma nv_diag_suppress 802

#include "flashinfer/attention/hopper/default_params.cuh"
#include "flashinfer/attention/hopper/prefill_sm90.cuh"
#include "flashinfer/attention/hopper/variants.cuh"

namespace flashinfer {

using Params = BatchPrefillPagedParams<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, int32_t>;

template <
    uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, MaskMode MASK_MODE, bool LEFT_SLIDING_WINDOW,
    bool SAME_SCHEDULE_FOR_ALL_HEADS, typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params &params, cudaStream_t stream);

template <bool use_logits_soft_cap>
using DefaultAttention = std::conditional_t<use_logits_soft_cap, LogitsSoftCap, StandardAttention>;

using AttentionVariant = DefaultAttention</*use_logits_soft_cap=*/false>;

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<
    128, 128, MaskMode::kCausal, false, false, AttentionVariant, Params>(Params &params, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<
    128, 128, MaskMode::kCausal, false, true, AttentionVariant, Params>(Params &params, cudaStream_t stream);

}  // namespace flashinfer

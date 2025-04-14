#include <cuda_bf16.h>
#include <fstream>

#include "attention.h"
#include "utility.h"

#include "flashinfer/attention/hopper/default_params.cuh"
#include "flashinfer/attention/hopper/variants.cuh"
#include "flashinfer/attention/scheduler.cuh"
#include "flashinfer/attention/mask.cuh"
#include "flashinfer/cutlass_utils.cuh"
#include "flashinfer/page.cuh"

#define _DISPATCH_SWITCH(var_name, cond, ...)                                                     \
    [&]() -> bool {                                                                               \
        switch (cond) {                                                                           \
            __VA_ARGS__                                                                           \
        default:                                                                                  \
            std::cerr << __PRETTY_FUNCTION__ << " failed to dispatch " var_name " " << int(cond); \
            assert(false);                                                                        \
            return false;                                                                         \
        }                                                                                         \
    }()

inline constexpr uint32_t pack_u16(uint16_t a, uint16_t b) {
    return (uint32_t(a) << 16) | uint32_t(b);
}

#define _DISPATCH_SWITCH_U16x2(var1_name, var2_name, cond1, cond2, ...)                                               \
    [&]() -> bool {                                                                                                   \
        switch (pack_u16(cond1, cond2)) {                                                                             \
            __VA_ARGS__                                                                                               \
        default:                                                                                                      \
            std::cerr << __PRETTY_FUNCTION__ << " failed to dispatch (" var1_name ", " var2_name "): (" << int(cond1) \
                      << ", " << int(cond2) << ")";                                                                   \
            assert(false);                                                                                            \
            return false;                                                                                             \
        }                                                                                                             \
    }()

#define _DISPATCH_CASE(case_expr, case_var, ...) \
    case case_expr:                              \
        {                                        \
            constexpr auto case_var = case_expr; \
            return __VA_ARGS__();                \
        }

#define _DISPATCH_CASE_U16x2(case_expr1, case_expr2, case_var1, case_var2, ...) \
    case pack_u16(case_expr1, case_expr2):                                      \
        {                                                                       \
            constexpr auto case_var1 = case_expr1;                              \
            constexpr auto case_var2 = case_expr2;                              \
            return __VA_ARGS__();                                               \
        }

#define DISPATCH_BOOL(expr, const_expr, ...)   \
    [&]() -> bool {                            \
        if (expr) {                            \
            constexpr bool const_expr = true;  \
            return __VA_ARGS__();              \
        } else {                               \
            constexpr bool const_expr = false; \
            return __VA_ARGS__();              \
        }                                      \
    }()

#define _DISPATCH_CASES_head_dim(case_var, ...) \
    _DISPATCH_CASE(128, case_var, __VA_ARGS__)  \
    // EOL

// clang-format off

#define _DISPATCH_CASES_head_dim_sm90(case_var1, case_var2, ...)      \
    _DISPATCH_CASE_U16x2(128, 128, case_var1, case_var2, __VA_ARGS__) \
    // EOL

// clang-format on

#define _DISPATCH_CASES_pos_encoding_mode(case_var, ...)          \
    _DISPATCH_CASE(PosEncodingMode::kNone, case_var, __VA_ARGS__) \
    // EOL

#define _DISPATCH_CASES_use_fp16_qk_reduction(case_var, ...) \
    _DISPATCH_CASE(false, case_var, __VA_ARGS__)             \
    // EOL

#define _DISPATCH_CASES_mask_mode(case_var, ...)             \
    _DISPATCH_CASE(MaskMode::kCausal, case_var, __VA_ARGS__) \
    // EOL

#define DISPATCH_head_dim(expr, const_expr, ...) \
    _DISPATCH_SWITCH("head_dim", expr, _DISPATCH_CASES_head_dim(const_expr, __VA_ARGS__))

#define DISPATCH_head_dim_sm90(expr1, expr2, const_expr1, const_expr2, ...) \
    _DISPATCH_SWITCH_U16x2(                                                 \
        "head_dim_qk", "head_dim_vo", expr1, expr2,                         \
        _DISPATCH_CASES_head_dim_sm90(const_expr1, const_expr2, __VA_ARGS__))

#define DISPATCH_pos_encoding_mode(expr, const_expr, ...) \
    _DISPATCH_SWITCH("positional encoding mode", expr, _DISPATCH_CASES_pos_encoding_mode(const_expr, __VA_ARGS__))

#define DISPATCH_use_fp16_qk_reduction(expr, const_expr, ...) \
    _DISPATCH_SWITCH("use_fp16_qk_reduction", expr, _DISPATCH_CASES_use_fp16_qk_reduction(const_expr, __VA_ARGS__))

#define DISPATCH_mask_mode(expr, const_expr, ...) \
    _DISPATCH_SWITCH("mask_mode", expr, _DISPATCH_CASES_mask_mode(const_expr, __VA_ARGS__))

#define DISPATCH_context(                                                                                \
    DTypeQ_, DTypeKV_, DTypeO_, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, AttentionVariant, PagedParams, ...) \
    {                                                                                                    \
        DISPATCH_mask_mode(mask_mode, MASK_MODE, [&] {                                                   \
            using DTypeQ_ = cutlass_dtype_t<DTypeQ>;                                                     \
            using DTypeKV_ = DTypeQ_;                                                                    \
            using DTypeO_ = DTypeQ_;                                                                     \
            using PagedParams = BatchPrefillPagedParams<DTypeQ_, DTypeKV_, DTypeO_, IdType>;             \
            return DISPATCH_head_dim_sm90(head_dim_qk, head_dim_vo, HEAD_DIM_QK, HEAD_DIM_VO, [&] {      \
                constexpr bool USE_SLIDING_WINDOW = false;                                               \
                constexpr bool USE_LOGITS_SOFT_CAP = false;                                              \
                using AttentionVariant = DefaultAttention<USE_LOGITS_SOFT_CAP>;                          \
                __VA_ARGS__();                                                                           \
                return true;                                                                             \
            });                                                                                          \
        });                                                                                              \
    }

namespace flashinfer {

template <
    uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, MaskMode MASK_MODE, bool LEFT_SLIDING_WINDOW,
    bool SAME_SCHEDULE_FOR_ALL_HEADS, typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params &params, cudaStream_t stream);

template <typename DTypeO, typename IdType>
std::vector<int64_t> BatchPrefillWithKVCacheSM90Plan(
    void *float_workspace_buffer, size_t float_workspace_size_in_bytes, void *int_workspace_buffer,
    void *page_locked_int_workspace_buffer, size_t int_workspace_size_in_bytes, IdType *qo_indptr, IdType *kv_indptr,
    IdType *kv_len_arr, uint32_t total_num_rows, uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t page_size, bool enable_cuda_graph, uint32_t head_dim_qk, uint32_t head_dim_vo, bool causal,
    cudaStream_t stream) {
    PrefillPlanSM90Info plan_info;

    cudaError_t status = PrefillSM90Plan(
        float_workspace_buffer, float_workspace_size_in_bytes, int_workspace_buffer, page_locked_int_workspace_buffer,
        int_workspace_size_in_bytes, plan_info, qo_indptr, kv_indptr, kv_len_arr, total_num_rows, batch_size,
        num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo, page_size, causal, enable_cuda_graph, sizeof(DTypeO),
        stream);
    assert(status == cudaSuccess);

    return plan_info.ToVector();
}

INSTANTIATE_FUNC_TU(BatchPrefillWithKVCacheSM90Plan, nv_bfloat16, int32_t);

template <typename DTypeQ, typename DTypeKV, typename DTypeO, typename IdType>
void BatchPrefillWithPagedKVCacheSM90Run(
    void *int_workspace_buffer, std::vector<int64_t> plan_info_vec, DTypeQ *q, paged_kv_t<DTypeKV, IdType> paged_kv,
    IdType *vector_sparse_indices, DTypeO *o, float *lse, uint32_t total_qolen, uint32_t num_qo_heads,
    uint32_t num_kv_heads, uint32_t head_dim_qk, uint32_t head_dim_vo, MaskMode mask_mode, cudaStream_t stream) {
    PrefillPlanSM90Info plan_info;
    plan_info.FromVector(plan_info_vec);

    constexpr int32_t window_left = -1;
    constexpr float logits_soft_cap = 0.f;

    const float sm_scale = 1.f / std::sqrt(float(head_dim_vo));

    DISPATCH_context(
        DTypeQ_, DTypeKV_, DTypeO_, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, AttentionVariant, PagedParams, [&] {
            PagedParams params;

            params.q_ptr = reinterpret_cast<DTypeQ_ *>(q);
            params.k_ptr = reinterpret_cast<DTypeKV_ *>(paged_kv.k_data);
            params.v_ptr = reinterpret_cast<DTypeKV_ *>(paged_kv.v_data);
            params.o_ptr = reinterpret_cast<DTypeO_ *>(o);
            params.lse_ptr = lse;

            params.q_stride_n = num_qo_heads * head_dim_qk;
            params.q_stride_h = head_dim_qk;
            params.o_stride_n = num_qo_heads * head_dim_vo;
            params.o_stride_h = head_dim_vo;

            params.k_stride_n = paged_kv.stride_n;
            params.k_stride_h = paged_kv.stride_h;
            params.v_stride_n = paged_kv.stride_n;
            params.v_stride_h = paged_kv.stride_h;

            params.nnz_qo = total_qolen;
            params.num_qo_heads = num_qo_heads;
            params.num_kv_heads = num_kv_heads;
            params.group_size = params.num_qo_heads / num_kv_heads;
            params.page_size = paged_kv.page_size;
            params.window_left = window_left;
            params.causal = mask_mode == MaskMode::kCausal;

            params.qo_tile_indices =
                GetPtrFromBaseOffset<IdType>(int_workspace_buffer, plan_info.qo_tile_indices_offset);
            params.qo_indptr = GetPtrFromBaseOffset<IdType>(int_workspace_buffer, plan_info.qo_indptr_offset);
            params.kv_indptr = GetPtrFromBaseOffset<IdType>(int_workspace_buffer, plan_info.kv_indptr_offset);
            params.qo_lens = GetPtrFromBaseOffset<IdType>(int_workspace_buffer, plan_info.qo_len_offset);
            params.kv_lens = GetPtrFromBaseOffset<IdType>(int_workspace_buffer, plan_info.kv_len_offset);
            params.head_indices = GetPtrFromBaseOffset<IdType>(int_workspace_buffer, plan_info.head_indices_offset);
            params.work_indptr = GetPtrFromBaseOffset<IdType>(int_workspace_buffer, plan_info.work_indptr_offset);

            params.kv_indices = vector_sparse_indices;

            params.additional_params.logits_soft_cap = logits_soft_cap;
            params.additional_params.sm_scale = sm_scale;

            bool same_schedule_for_all_heads = plan_info.same_schedule_for_all_heads;
            DISPATCH_BOOL(same_schedule_for_all_heads, SAME_SCHEDULER_FOR_ALL_HEADS, [&] {
                cudaError_t status = BatchPrefillWithPagedKVCacheDispatched<
                    HEAD_DIM_QK, HEAD_DIM_VO, MASK_MODE, USE_SLIDING_WINDOW, SAME_SCHEDULER_FOR_ALL_HEADS,
                    AttentionVariant>(params, stream);
                assert(status == cudaSuccess);
                return true;
            });
        });
}

INSTANTIATE_FUNC_TUVW(BatchPrefillWithPagedKVCacheSM90Run, nv_bfloat16, nv_bfloat16, nv_bfloat16, int32_t);

}  // namespace flashinfer

template <typename T>
void RunAttentionSm90(
    cudaStream_t stream, void *float_workspace, size_t float_workspace_size, void *int_workspace,
    void *pinned_int_workspace, size_t int_workspace_size, T *output, T *q_ptr, T *k_ptr, T *v_ptr, T *k_cache,
    T *v_cache, int32_t *qo_offset_host, int32_t *kv_offset_host, int32_t *kvlens_host, int32_t *kv_offset,
    int32_t *kvlens, int32_t *batch_indices, int32_t *positions, int32_t *kv_indices, int32_t *kv_indptr,
    int32_t *kv_last_page_len, int32_t *vector_sparse_indices, int32_t batch_size, int32_t total_qolen,
    int32_t page_size, int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim, size_t max_pos_embedding,
    bool is_causal = true, float *softmax_lse = nullptr) {
    // Each layer's KV cache layout: [num_pages, page_size, num_heads, head_dim]
    const int64_t stride_h = head_dim;
    const int64_t stride_n = num_kv_heads * stride_h;
    const int64_t stride_page = page_size * stride_n;

    ::flashinfer::paged_kv_t<T, int32_t> paged_kv(
        num_kv_heads, page_size, head_dim, batch_size, ::flashinfer::QKVLayout::kNHD, k_cache, v_cache, kv_indices,
        kv_indptr, kv_last_page_len);

    cudaError_t status = ::flashinfer::AppendPagedKVCache(
        paged_kv, k_ptr, v_ptr, batch_indices, positions, total_qolen, stride_n, stride_h, stride_n, stride_h, stream);
    assert(status == cudaSuccess);

    status = ::flashinfer::BlockSparseIndicesToVectorSparseOffset(
        kv_indices, kv_indptr, vector_sparse_indices, kv_offset, kvlens, stride_page / stride_n, 1, batch_size,
        page_size, stream);
    assert(status == cudaSuccess);

    std::vector<int64_t> plan_info_vec;

    plan_info_vec = ::flashinfer::BatchPrefillWithKVCacheSM90Plan<T, int32_t>(
        float_workspace, float_workspace_size, int_workspace, pinned_int_workspace, int_workspace_size, qo_offset_host,
        kv_offset_host, kvlens_host, total_qolen, batch_size, num_qo_heads, num_kv_heads, page_size, false, head_dim,
        head_dim, is_causal, stream);

    ::flashinfer::BatchPrefillWithPagedKVCacheSM90Run<T, T, T, int32_t>(
        int_workspace, plan_info_vec, q_ptr, paged_kv, vector_sparse_indices, output, softmax_lse, total_qolen,
        num_qo_heads, num_kv_heads, head_dim, head_dim, ::flashinfer::MaskMode::kCausal, stream);

    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();

    {
        std::vector<T> output_vec(total_qolen * num_qo_heads * head_dim);
        cudaMemcpyAsync(
            output_vec.data(), output, total_qolen * num_qo_heads * head_dim * sizeof(T), cudaMemcpyDeviceToHost,
            stream);
        cudaStreamSynchronize(stream);

        // Check if output contains NaN values
        bool has_nan = false;
        for (size_t i = 0; i < output_vec.size(); ++i) {
            if (__hisnan(output_vec[i])) {
                has_nan = true;
                std::cerr << "NaN detected in output at index " << i << " in max_pos_emeding at : " << max_pos_embedding << std::endl;
                break;
            }
        }

        if (has_nan) {
            std::cerr << "Output contains NaN values after BatchPrefillWithPagedKVCacheSM90Run" << std::endl;
            std::cerr << "batch_size: " << batch_size << ", total_qolen: " << total_qolen
                      << ", num_qo_heads: " << num_qo_heads << ", num_kv_heads: " << num_kv_heads
                      << ", head_dim: " << head_dim << ", page_size: " << page_size << std::endl;

            std::ofstream ofs("o.txt");
            for (int i = 0; i < total_qolen; ++i) {
                for (int j = 0; j < num_qo_heads; ++j) {
                    ofs << "i: " << i << ", j: " << j << std::endl;
                    for (int k = 0; k < head_dim; ++k) {
                        ofs << float(output_vec[i * num_qo_heads * head_dim + j * head_dim + k]) << " ";
                        if ((k + 1) % 8 == 0)
                            ofs << std::endl;
                    }
                    ofs << std::endl;
                }
                ofs << std::endl;
            }

            assert(false);
        }
    }
}

INSTANTIATE_FUNC_T(RunAttentionSm90, nv_bfloat16);

template <typename T>
std::vector<T> single_mha(
    const std::vector<T> &q, const std::vector<T> &k, const std::vector<T> &v, size_t qo_len, size_t kv_len,
    size_t num_qo_heads, size_t num_kv_heads, size_t head_dim, bool causal = true, uint8_t *custom_mask = nullptr) {
    assert(qo_len <= kv_len);
    assert(num_qo_heads % num_kv_heads == 0);

    float sm_scale = 1.f / std::sqrt(float(head_dim));
    std::vector<T> o(qo_len * num_qo_heads * head_dim);
    std::vector<float> att(kv_len);

    ::flashinfer::tensor_info_t info(
        qo_len, kv_len, num_qo_heads, num_kv_heads, ::flashinfer::QKVLayout::kNHD, head_dim);

    for (size_t qo_head_idx = 0; qo_head_idx < num_qo_heads; ++qo_head_idx) {
        const size_t kv_head_idx = qo_head_idx / info.get_group_size();
        for (size_t qo_idx = 0; qo_idx < qo_len; ++qo_idx) {
            float max_val = -5e4;
            for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
                att[kv_idx] = 0.;
                for (size_t feat_idx = 0; feat_idx < head_dim; ++feat_idx) {
                    att[kv_idx] += float(q[info.get_q_elem_offset(qo_idx, qo_head_idx, feat_idx)])
                                   * float(k[info.get_kv_elem_offset(kv_idx, kv_head_idx, feat_idx)]) * sm_scale;
                }

                // apply mask
                if (custom_mask) {
                    constexpr size_t uint8_bits = std::numeric_limits<uint8_t>::digits;
                    const size_t idx = qo_idx * kv_len + kv_idx;
                    auto [pack_idx, bit_idx] = std::make_tuple(idx / uint8_bits, idx % uint8_bits);
                    if ((custom_mask[pack_idx] & (1 << bit_idx)) == 0) {
                        att[kv_idx] = -5e4;
                    }
                } else if (causal) {
                    if (kv_idx > kv_len + qo_idx - qo_len) {
                        att[kv_idx] = -5e4;
                    }
                }
                max_val = std::max(max_val, att[kv_idx]);
            }
            // exp minus max
            float denom = 0;
            for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
                att[kv_idx] = std::exp(att[kv_idx] - max_val);
                denom += att[kv_idx];
            }

            // divide by denom
            for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
                att[kv_idx] /= denom;
            }

            for (size_t feat_idx = 0; feat_idx < head_dim; ++feat_idx) {
                float o_float = 0.;
                for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
                    o_float += att[kv_idx] * float(v[info.get_kv_elem_offset(kv_idx, kv_head_idx, feat_idx)]);
                }
                o[info.get_o_elem_offset(qo_idx, qo_head_idx, feat_idx)] = T(o_float);
            }
        }
    }
    return o;
}

template <typename T>
std::vector<T> MultiHeadAttentionRef(
    const std::vector<T> &q, const std::vector<T> &k, const std::vector<T> &v,
    const std::vector<int32_t> &qo_offset_host, const std::vector<int32_t> &kv_offset_host, size_t total_qolen,
    size_t batch_size, size_t num_qo_heads, size_t num_kv_heads, size_t head_dim, bool causal = true,
    uint8_t *custom_mask = nullptr) {
    std::vector<T> o(total_qolen * num_qo_heads * head_dim);
    for (size_t b0 = 0; b0 < batch_size; ++b0) {
        size_t qo_len = static_cast<size_t>(qo_offset_host[b0 + 1] - qo_offset_host[b0]);
        size_t kv_len = static_cast<size_t>(kv_offset_host[b0 + 1] - kv_offset_host[b0]);

        size_t qo_offset = size_t(qo_offset_host[b0]) * num_qo_heads * head_dim;
        size_t kv_offset = size_t(kv_offset_host[b0]) * num_kv_heads * head_dim;

        std::vector<T> single_q(q.begin() + qo_offset, q.begin() + qo_offset + qo_len * num_qo_heads * head_dim);
        std::vector<T> single_k(k.begin() + kv_offset, k.begin() + kv_offset + kv_len * num_kv_heads * head_dim);
        std::vector<T> single_v(v.begin() + kv_offset, v.begin() + kv_offset + kv_len * num_kv_heads * head_dim);

        std::vector<T> single_o = single_mha(
            single_q, single_k, single_v, qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim, causal, custom_mask);

        std::copy(single_o.begin(), single_o.end(), o.begin() + qo_offset);

        if (custom_mask) {
            constexpr size_t uint8_bits = std::numeric_limits<uint8_t>::digits;
            const size_t offset = (qo_len * kv_len + uint8_bits - 1) / uint8_bits;
            custom_mask += offset;
        }
    }
    return o;
}

template <typename T>
void Attention(cudaStream_t stream, size_t max_pos_embedding) {
    constexpr int32_t max_num_pages = 1024;
    constexpr int32_t page_size = 1;
    constexpr int32_t num_qo_heads = 28;
    constexpr int32_t num_kv_heads = 4;
    constexpr int32_t head_dim = 128;
    int32_t batch_size = 1;

    const float scale_factor = 1.f / static_cast<float>(head_dim);
    // const float scale_factor = 1.f;

    // [max_num_pages, page_size, num_kv_heads, head_dim]
    T *k_blocks = nullptr;
    T *v_blocks = nullptr;
    cudaMallocAsync(&k_blocks, max_num_pages * page_size * num_kv_heads * head_dim * sizeof(T), stream);
    cudaMallocAsync(&v_blocks, max_num_pages * page_size * num_kv_heads * head_dim * sizeof(T), stream);
    cudaStreamSynchronize(stream);

    int32_t block_id = 0;
    std::vector<T> flatten_q_host;
    std::vector<T> flatten_k_host;
    std::vector<T> flatten_v_host;

    std::vector<T> flatten_k_all_host;
    std::vector<T> flatten_v_all_host;

    std::vector<int32_t> kvlens_host;
    std::vector<int32_t> qo_offset_host;
    std::vector<int32_t> kv_offset_host;
    std::vector<int32_t> kv_last_page_len_host;

    std::vector<int32_t> paged_kv_indices_host;
    std::vector<int32_t> paged_kv_indptr_host;
    std::vector<int32_t> batch_indices_host;
    std::vector<int32_t> positions_host;

    qo_offset_host = {0};
    kv_offset_host = {0};

    paged_kv_indptr_host = {0};

    int32_t total_qolen = 0;

    for (size_t i = 0; i < batch_size; ++i) {
        const int32_t qo_len = 1;
        const int32_t kv_len = 1;

        const int32_t num_pages = DivideUp(kv_len, page_size);
        const int32_t kv_last_page_len = (kv_len - 1) % page_size + 1;

        std::vector<T> qi(qo_len * num_qo_heads * head_dim);
        std::vector<T> ki(kv_len * num_kv_heads * head_dim);
        std::vector<T> vi(kv_len * num_kv_heads * head_dim);

        RandomGenerator::GenNorm<T>(qi, scale_factor);
        RandomGenerator::GenNorm<T>(ki, scale_factor);
        RandomGenerator::GenNorm<T>(vi, scale_factor);

        std::vector<int64_t> block_ids;
        std::vector<void *> k_block_addrs;
        std::vector<void *> v_block_addrs;
        for (int32_t j = 0; j < num_pages; ++j) {
            block_ids.push_back(block_id++);
            k_block_addrs.push_back(k_blocks + j * page_size * num_kv_heads * head_dim);
            v_block_addrs.push_back(v_blocks + j * page_size * num_kv_heads * head_dim);
        }

        std::vector<int32_t> kv_indices;
        for (int64_t block_id : block_ids) {
            kv_indices.push_back(block_id);
        }

        const int32_t cache_len = kv_len - qo_len;
        const int32_t num_cache_pages = DivideUp(cache_len, page_size);
        const int32_t cache_last_page_len = (cache_len - 1) % page_size + 1;
        for (int32_t j = 0; j < num_cache_pages; ++j) {
            const int32_t host_offset = j * page_size * num_kv_heads * head_dim;
            T *k_cache_ptr = static_cast<T *>(k_block_addrs[j]);
            T *v_cache_ptr = static_cast<T *>(v_block_addrs[j]);
            const int32_t cache_len_j = (j == num_cache_pages - 1) ? cache_last_page_len : page_size;
            const int32_t elems = cache_len_j * num_kv_heads * head_dim;

            cudaMemcpyAsync(
                k_cache_ptr, ki.data() + host_offset, size_t(elems) * sizeof(T), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(
                v_cache_ptr, vi.data() + host_offset, size_t(elems) * sizeof(T), cudaMemcpyHostToDevice, stream);
        }

        flatten_q_host.insert(flatten_q_host.end(), qi.begin(), qi.end());
        flatten_k_host.insert(flatten_k_host.end(), ki.begin() + cache_len * num_kv_heads * head_dim, ki.end());
        flatten_v_host.insert(flatten_v_host.end(), vi.begin() + cache_len * num_kv_heads * head_dim, vi.end());

        flatten_k_all_host.insert(flatten_k_all_host.end(), ki.begin(), ki.end());
        flatten_v_all_host.insert(flatten_v_all_host.end(), vi.begin(), vi.end());

        total_qolen += qo_len;

        kvlens_host.push_back(kv_len);
        qo_offset_host.push_back(qo_offset_host.back() + qo_len);
        kv_offset_host.push_back(kv_offset_host.back() + kv_len);
        kv_last_page_len_host.push_back(kv_last_page_len);

        paged_kv_indptr_host.push_back(paged_kv_indptr_host.back() + num_pages);
        paged_kv_indices_host.insert(paged_kv_indices_host.end(), kv_indices.begin(), kv_indices.end());
        for (int32_t j = 0; j < qo_len; ++j) {
            batch_indices_host.push_back(i);
            const int32_t pos = kv_len - qo_len + j;
            positions_host.push_back(pos);
        }
    }

    std::vector<int64_t> qo_shape = {total_qolen, num_qo_heads, head_dim};
    std::vector<int64_t> kv_shape = {total_qolen, num_kv_heads, head_dim};

    T *q_tensor = nullptr;
    T *k_tensor = nullptr;
    T *v_tensor = nullptr;
    T *o_tensor = nullptr;

    int32_t *kvlens_tensor = nullptr;
    int32_t *kv_offset_tensor = nullptr;
    int32_t *kv_last_page_len_tensor = nullptr;

    int32_t *paged_kv_indices_tensor = nullptr;
    int32_t *paged_kv_indptr_tensor = nullptr;

    int32_t *batch_indices_tensor = nullptr;
    int32_t *positions_tensor = nullptr;
    int32_t *vector_sparse_indices_tensor = nullptr;

    cudaMallocAsync(&q_tensor, total_qolen * num_qo_heads * head_dim * sizeof(T), stream);
    cudaMallocAsync(&k_tensor, total_qolen * num_kv_heads * head_dim * sizeof(T), stream);
    cudaMallocAsync(&v_tensor, total_qolen * num_kv_heads * head_dim * sizeof(T), stream);
    cudaMallocAsync(&o_tensor, total_qolen * num_qo_heads * head_dim * sizeof(T), stream);

    cudaMallocAsync(&kvlens_tensor, batch_size * sizeof(int32_t), stream);
    cudaMallocAsync(&kv_offset_tensor, (batch_size + 1) * sizeof(int32_t), stream);
    cudaMallocAsync(&kv_last_page_len_tensor, batch_size * sizeof(int32_t), stream);

    cudaMallocAsync(&paged_kv_indices_tensor, paged_kv_indptr_host.back() * sizeof(int32_t), stream);
    cudaMallocAsync(&paged_kv_indptr_tensor, (batch_size + 1) * sizeof(int32_t), stream);

    cudaMallocAsync(&batch_indices_tensor, total_qolen * sizeof(int32_t), stream);
    cudaMallocAsync(&positions_tensor, total_qolen * sizeof(int32_t), stream);
    cudaMallocAsync(&vector_sparse_indices_tensor, kv_offset_host.back() * sizeof(int32_t), stream);
    cudaStreamSynchronize(stream);

    cudaMemcpyAsync(
        q_tensor, flatten_q_host.data(), total_qolen * num_qo_heads * head_dim * sizeof(T), cudaMemcpyHostToDevice,
        stream);
    cudaMemcpyAsync(
        k_tensor, flatten_k_host.data(), total_qolen * num_kv_heads * head_dim * sizeof(T), cudaMemcpyHostToDevice,
        stream);
    cudaMemcpyAsync(
        v_tensor, flatten_v_host.data(), total_qolen * num_kv_heads * head_dim * sizeof(T), cudaMemcpyHostToDevice,
        stream);
    cudaMemsetAsync(o_tensor, 0, total_qolen * num_qo_heads * head_dim * sizeof(T), stream);

    cudaMemcpyAsync(kvlens_tensor, kvlens_host.data(), batch_size * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(
        kv_offset_tensor, kv_offset_host.data(), (batch_size + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(
        kv_last_page_len_tensor, kv_last_page_len_host.data(), batch_size * sizeof(int32_t), cudaMemcpyHostToDevice,
        stream);

    cudaMemcpyAsync(
        paged_kv_indices_tensor, paged_kv_indices_host.data(), paged_kv_indptr_host.back() * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(
        paged_kv_indptr_tensor, paged_kv_indptr_host.data(), (batch_size + 1) * sizeof(int32_t), cudaMemcpyHostToDevice,
        stream);
    cudaMemcpyAsync(
        batch_indices_tensor, batch_indices_host.data(), total_qolen * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(
        positions_tensor, positions_host.data(), total_qolen * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

    cudaMemsetAsync(vector_sparse_indices_tensor, 0, kv_offset_host.back() * sizeof(int32_t), stream);

    void *float_workspace = nullptr;
    void *int_workspace = nullptr;
    void *pinned_int_workspace = nullptr;

    size_t float_workspace_size = 512 * 1024 * 1024;
    size_t int_workspace_size = 512 * 1024 * 1024;

    cudaMallocAsync(&float_workspace, float_workspace_size, stream);
    cudaMallocAsync(&int_workspace, int_workspace_size, stream);
    cudaMallocHost(&pinned_int_workspace, int_workspace_size);
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();

    RunAttentionSm90(
        stream, float_workspace, float_workspace_size, int_workspace, pinned_int_workspace, int_workspace_size,
        o_tensor, q_tensor, k_tensor, v_tensor, k_blocks, v_blocks, qo_offset_host.data(), kv_offset_host.data(),
        kvlens_host.data(), kv_offset_tensor, kvlens_tensor, batch_indices_tensor, positions_tensor,
        paged_kv_indices_tensor, paged_kv_indptr_tensor, kv_last_page_len_tensor, vector_sparse_indices_tensor,
        batch_size, total_qolen, page_size, num_qo_heads, num_kv_heads, head_dim, max_pos_embedding);

    // check_for_nan_in_output<T>(o_tensor, total_qolen * num_qo_heads * head_dim, stream, "Attention");

    std::vector<T> o_host(total_qolen * num_qo_heads * head_dim);
    cudaMemcpyAsync(
        o_host.data(), o_tensor, total_qolen * num_qo_heads * head_dim * sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<T> o_ref = MultiHeadAttentionRef(
        flatten_q_host, flatten_k_all_host, flatten_v_all_host, qo_offset_host, kv_offset_host, total_qolen, batch_size,
        num_qo_heads, num_kv_heads, head_dim);

    double rtol = 1.6e-2, atol = 2e-3;
    for (size_t i = 0; i < o_host.size(); ++i) {
        const float o_host_val = static_cast<float>(o_host[i]);
        const float o_ref_val = static_cast<float>(o_ref[i]);
        if (std::abs(o_host_val - o_ref_val) > atol && std::abs(o_host_val - o_ref_val) / std::abs(o_ref_val) > rtol) {
            std::cout << "o_host[" << i << "] = " << o_host_val << ", o_ref[" << i << "] = " << o_ref_val << std::endl;
        }
    }

    cudaFreeAsync(k_blocks, stream);
    cudaFreeAsync(v_blocks, stream);

    cudaFreeAsync(q_tensor, stream);
    cudaFreeAsync(k_tensor, stream);
    cudaFreeAsync(v_tensor, stream);
    cudaFreeAsync(o_tensor, stream);

    cudaFreeAsync(kvlens_tensor, stream);
    cudaFreeAsync(kv_offset_tensor, stream);
    cudaFreeAsync(kv_last_page_len_tensor, stream);

    cudaFreeAsync(paged_kv_indices_tensor, stream);
    cudaFreeAsync(paged_kv_indptr_tensor, stream);

    cudaFreeAsync(batch_indices_tensor, stream);
    cudaFreeAsync(positions_tensor, stream);
    cudaFreeAsync(vector_sparse_indices_tensor, stream);

    cudaFreeAsync(float_workspace, stream);
    cudaFreeAsync(int_workspace, stream);
    cudaFreeHost(pinned_int_workspace);
}

INSTANTIATE_FUNC_T(Attention, nv_bfloat16);

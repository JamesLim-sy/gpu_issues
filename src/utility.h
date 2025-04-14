#pragma once

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

namespace details {

template <typename DstType, typename SrcType>
inline DstType TypeCast(SrcType val) {
    if constexpr (std::is_same_v<SrcType, DstType>) {
        return val;
    } else {
        return static_cast<DstType>(val);
    }
}

}  // namespace details

template <typename T, typename U>
inline std::common_type_t<T, U> DivideUp(T n, U d) {
    static_assert(std::is_integral<T>::value, "T must be an integral type");
    static_assert(std::is_integral<U>::value, "U must be an integral type");

    using W = std::common_type_t<T, U>;
    W n_ = details::TypeCast<W>(n);
    W d_ = details::TypeCast<W>(d);

    return (n_ + d_ - 1) / d_;
}

class RandomGenerator {
   public:
    template <typename T>
    static void GenNorm(
        std::vector<T> &vec, float scale_factor = 1.0f, float min_val = -1.0f, float max_val = 1.0f, float mean = 0.f,
        float stddev = 1.f) {
        static std::mt19937 gen;
        std::normal_distribution<float> d{mean, stddev};
        for (size_t i = 0; i < vec.size(); ++i) {
            float raw_float = d(gen);
            vec[i] = T(std::clamp(raw_float, min_val, max_val) * scale_factor);
        }
    }
};

template <typename T>
void check_for_nan_in_output(const T* output, size_t numel, cudaStream_t stream, std::string error_msg) {
    std::vector<T> output_vec(numel);
    cudaMemcpyAsync(
        output_vec.data(), output, numel * sizeof(T),
        cudaMemcpyDeviceToHost,
        stream);
    cudaStreamSynchronize(stream);

    bool has_nan = false;
    for (size_t i = 0; i < output_vec.size(); ++i) {
        if (__hisnan(output_vec[i])) {
            has_nan = true;
            std::cerr << error_msg << " : NaN detected in output at index " << std::to_string(i) << std::endl;
            break;
        }
    }
}

#define INSTANTIATE_FUNC_T(func, T)             template decltype(func<T>) func<T>
#define INSTANTIATE_FUNC_TU(func, T, U)         template decltype(func<T, U>) func<T, U>
#define INSTANTIATE_FUNC_TUV(func, T, U, V)     template decltype(func<T, U, V>) func<T, U, V>
#define INSTANTIATE_FUNC_TUVW(func, T, U, V, W) template decltype(func<T, U, V, W>) func<T, U, V, W>

#define INSTANTIATE_CLASS_T(cls, T)             template class cls<T>
#define INSTANTIATE_CLASS_TU(cls, T, U)         template class cls<T, U>
#define INSTANTIATE_CLASS_TUV(cls, T, U, V)     template class cls<T, U, V>
#define INSTANTIATE_CLASS_TUVW(cls, T, U, V, W) template class cls<T, U, V, W>

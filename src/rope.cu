#include <exception>
#include <stdexcept>

#include "rope.h"
#include "utility.h"

template <typename T>
__global__ void InvokeComputeCosSinWithCacheKernel(
                T *__restrict__ cos_sin_cache, const int max_pos_embedding, const int rotary_dim, 
                const float base, const float scaling) {
    for (int pos = blockIdx.x; pos < max_pos_embedding; pos += gridDim.x) {
        for (int rid = threadIdx.x; rid < rotary_dim / 2; rid += blockDim.x) {
            float inv_freq = 1.0f / powf(base, rid * 2 / static_cast<float>(rotary_dim));
            float freq = pos * inv_freq / scaling;

            float cos_val, sin_val;
            sincosf(freq, &sin_val, &cos_val);
            cos_sin_cache[pos * rotary_dim + rid] = static_cast<T>(cos_val);
            cos_sin_cache[pos * rotary_dim + rotary_dim / 2 + rid] = static_cast<T>(sin_val);
        }
    }
}

void ComputeCosSinWithCache(Tensor& cos_sin_cache_tensor, cudaStream_t stream) {
    int64_t max_pos_embedding = cos_sin_cache_tensor.GetShape()[0];
    int64_t head_dim = cos_sin_cache_tensor.GetShape()[1];

    int num_sms = 0;
    int dev_id = 0;
    cudaGetDevice(&dev_id);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id);

    dim3 grid(num_sms);
    dim3 block(head_dim);

    float base = 10000.0f;
    float scaling = 1.0f;

    DISPATCH_DATATYPE(cos_sin_cache_tensor.GetDataType(),
        InvokeComputeCosSinWithCacheKernel<T><<<grid, block, 0, stream>>>(
            static_cast<T*>(cos_sin_cache_tensor.GetData()), max_pos_embedding, head_dim, base, scaling);
        cudaStreamSynchronize(stream);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
        }
        check_for_nan_in_output<T>(static_cast<T*>(cos_sin_cache_tensor.GetData()),
                                   cos_sin_cache_tensor.GetNumel(), stream,
                                   "ComputeCosSinWithCache");
    )
}

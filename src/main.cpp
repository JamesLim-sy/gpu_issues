#include <vector>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "attention.h"
#include "rope.h"
#include "tensor.h"

int main() {
    constexpr int num_stream = 2;
    cudaSetDevice(0);
    std::vector<cudaStream_t> streams(num_stream);
    for (int i = 0; i < num_stream; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    size_t max_pos_embedding_limit = 524288; // 512 * 1024

    Tensor temp_tensor({150 * 1024}, DataType::kBFloat16, /*is_cuda=*/true);
    InitTensorWithRandomValue(temp_tensor, streams[0], /*init_with_zero=*/true);

    for (auto max_pos_embedding = 32768; 
         max_pos_embedding <= max_pos_embedding_limit;
         max_pos_embedding *= 2) {
        Tensor cos_sin_cache_tensor({max_pos_embedding, 128}, DataType::kBFloat16, /*is_cuda=*/true);
        InitTensorWithRandomValue(cos_sin_cache_tensor, streams[0], /*init_with_zero=*/true);
        ComputeCosSinWithCache(cos_sin_cache_tensor, streams[0]);

        // cudaMemsetAsync(cos_sin_cache_tensor.GetData(), 0,
        //                 cos_sin_cache_tensor.GetTotalByteSize(), streams[0]);
        
        // NOTE: max_pos_embedding is just a mark to logout at which value
        //  nan happens in attention, it is not used in the attention kernel.
        Attention<nv_bfloat16>(streams[0], max_pos_embedding);
    }

    for (auto &stream : streams) {
        cudaStreamDestroy(stream);
    }
    return 0;
}

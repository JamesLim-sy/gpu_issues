#pragma once

#include <cuda_bf16.h>
#include "tensor.h"

void ComputeCosSinWithCache(Tensor& cos_sin_cache_tensor, cudaStream_t stream);

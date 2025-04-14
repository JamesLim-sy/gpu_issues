#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
void Attention(cudaStream_t stream, size_t max_pos_embedding);

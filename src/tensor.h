#pragma once

#include <vector>
#include <cuda_runtime.h>

#include "type.h"
#include "utility.h"

class Tensor {
 public:
    Tensor() = default;
    
    Tensor(std::vector<int64_t> shape, DataType dtype, bool is_cuda = true)
        : shape_(std::move(shape)), dtype_(dtype), is_cuda_(is_cuda) {}
    
    ~Tensor() {
        if (is_cuda_) {
            cudaFreeAsync(data_, nullptr);
        } else {
            delete[] static_cast<char*>(data_);
        }
    }
    
    Tensor(const Tensor &other) = delete;
    Tensor &operator=(const Tensor &other) = delete;
    
    void SetData(void *data) { data_ = data; }
    void *GetData() const { return data_; }
    void *GetData() { return data_;}
    
    size_t GetNumel() const {
        size_t numel_ = 1;
        for (const auto &dim : shape_) {
            numel_ *= dim;
        }
        return numel_;
    }
    
    size_t GetTotalByteSize() const {
        return GetNumel() * GetDataTypeSize(dtype_);
    }

    DataType GetDataType() const { return dtype_; }

    std::vector<int64_t> GetShape() const { return shape_; }

    bool IsCuda() const { return is_cuda_; }

   public:
    std::vector<int64_t> shape_;
    DataType dtype_;
    void *data_;
    bool is_cuda_;
};

inline void InitTensorWithRandomValue(Tensor& tensor, cudaStream_t stream, bool is_zero = false) {
    size_t numel = tensor.GetNumel();
    size_t total_byte_size = tensor.GetTotalByteSize();
    if (numel == 0) {
        throw std::invalid_argument("Tensor has zero elements");
    }

    DISPATCH_DATATYPE(tensor.GetDataType(),
        std::vector<T> host_data(numel, T(0));
        if (!is_zero) {
            RandomGenerator::GenNorm(host_data, 1.0f, -1.0f, 1.0f);
        }

        void* data = nullptr;
        if (tensor.IsCuda()) {
            cudaMallocAsync(&data, total_byte_size, stream);
            cudaMemcpyAsync(data, host_data.data(), total_byte_size, cudaMemcpyHostToDevice, stream);
            tensor.SetData(data);
            cudaStreamSynchronize(stream);
        } else {
            data = new char[total_byte_size];
            std::memcpy(data, host_data.data(), total_byte_size);
            tensor.SetData(data);
        }
    );
}
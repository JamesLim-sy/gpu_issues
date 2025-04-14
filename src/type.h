#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <memory>
#include <cstring>
#include <stdexcept>

enum class DataType {
    kFloat32 = 0,
    kFloat16 = 1,
    kBFloat16 = 2,
    kInt8 = 3,
    kInt32 = 4,
    kInt64 = 5,
    kBool = 6,
};

inline size_t GetDataTypeSize(DataType dtype) {
    switch (dtype) {
        case DataType::kFloat32:
            return sizeof(float);
        case DataType::kFloat16:
            return sizeof(__half);
        case DataType::kBFloat16:
            return sizeof(__nv_bfloat16);
        case DataType::kInt8:
            return sizeof(int8_t);
        case DataType::kInt32:
            return sizeof(int32_t);
        case DataType::kInt64:
            return sizeof(int64_t);
        case DataType::kBool:
            return sizeof(bool);
        default:
            throw std::invalid_argument("Unsupported data type");
    }
}

#define DISPATCH_DATA_TYPE_CASE(data_type, real_type, ...) \
    case data_type: { \
        using T = real_type; \
        __VA_ARGS__; \
        break; \
    }

#define DISPATCH_DATATYPE(dtype, ...) \
    switch (dtype) { \
        DISPATCH_DATA_TYPE_CASE(DataType::kFloat16, __half, __VA_ARGS__) \
        DISPATCH_DATA_TYPE_CASE(DataType::kBFloat16, __nv_bfloat16, __VA_ARGS__) \
        default: \
            throw std::runtime_error("Unsupported data type: " + std::to_string(static_cast<int>(dtype))); \
    }

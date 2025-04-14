# ThirdParty version
```bash
- cutlass: v3.7.0
- flashinfer: v0.2.4
```

## Install
```bash
# For CUDA 12.6 & torch 2.6
git submodule update --init
mkdir -p build
cp CMakeLists.txt build/ && cd build/
cmake .. && make -j
```

## Test
```bash
./main
```

## Explanation
```bash
Once run this demo, attention kernel will gen nan value like below, but if just run attention kernel or just rope kernel respectively, it will not emerge any nan value. However, the result calculated by rope kernel has no influence on attention kernel in the demo. More secifically, these 2 kernels just has no influence on each other. And once put a cuda memset kernel between them, the nan value will not emerge. Please help check why nan happens when running these 2 kernels together.

various combinations:
1. run rope kernel                                     -> no nan
2. run attention kernel                                -> no nan
3. run rope kernel + run attention kernel              -> nan value
4. run rope kernel + cudaMemset + run attention kernel -> no nan
```

## Emerging error example
```bash
NaN detected in output at index 402 in max_pos_emeding at : 131072
Output contains NaN values after BatchPrefillWithPagedKVCacheSM90Run
batch_size: 1, total_qolen: 1, num_qo_heads: 28, num_kv_heads: 4, head_dim: 128, page_size: 1
main: /xx/gpu_issues/src/attention.cu:302: void RunAttentionSm90(cudaStream_t, void*, size_t, void*, void*, size_t, T*, T*, T*, T*, T*, T*, int32_t*, int32_t*, int32_t*, int32_t*, int32_t*, int32_t*, int32_t*, int32_t*, int32_t*, int32_t*, int32_t*, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, size_t, bool, float*) [with T = __nv_bfloat16; cudaStream_t = CUstream_st*; size_t = long unsigned int; int32_t = int]: Assertion `false' failed.
Aborted
```

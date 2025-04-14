set(__SRCS__)
set(__INCLUDE_DIRS__)
set(__LINK_LIBS__)

file(GLOB __TMP_SRCS__
    *.cpp
    *.cu)
list(APPEND __SRCS__ ${__TMP_SRCS__})
unset(__TMP_SRCS__)

list(APPEND __INCLUDE_DIRS__
    ${CUDAToolkit_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
    ${PROJECT_SOURCE_DIR}/third_party/cutlass/tools/util/include
    ${PROJECT_SOURCE_DIR}/third_party/flashinfer/include)
list(APPEND __LINK_LIBS__ CUDA::cuda_driver CUDA::cudart)

add_executable(main ${__SRCS__})

target_include_directories(main PUBLIC
    ${__INCLUDE_DIRS__})

target_link_libraries(main PRIVATE
    ${__LINK_LIBS__})

unset(__SRCS__)
unset(__INCLUDE_DIRS__)
unset(__LINK_LIBS__)

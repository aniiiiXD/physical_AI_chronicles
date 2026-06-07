cmake_minimum_required(VERSION 3.20)
project(cuda_kernels CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

# RTX 3060 = Ampere = sm_86
# Compile native code for 86, also embed PTX for forward-compat
set(CMAKE_CUDA_ARCHITECTURES 86)

# vec_add — no external deps
add_executable(vec_add vec_add.cu)
target_compile_options(vec_add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    -Xcompiler -O3
>)
target_link_libraries(vec_add PRIVATE cudart)

# matmul — links cuBLAS
find_package(CUDAToolkit REQUIRED)
add_executable(matmul matmul.cu)
target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    -Xcompiler -O3
>)
target_link_libraries(matmul PRIVATE cudart CUDA::cublas)

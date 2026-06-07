cmake_minimum_required(VERSION 3.15)
project(simd_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(simd_bench simd_bench.cpp)
target_compile_options(simd_bench PRIVATE -O2 -Wall -Wextra)

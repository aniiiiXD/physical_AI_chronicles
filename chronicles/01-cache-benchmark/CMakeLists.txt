cmake_minimum_required(VERSION 3.15)

project(cache_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cache_benchmark cache_benchmark.cpp)

target_compile_options(cache_benchmark PRIVATE -O2 -Wall -Wextra)

cmake_minimum_required(VERSION 3.15)
project(matmul LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)
target_compile_options(matmul PRIVATE -O2 -Wall -Wextra)

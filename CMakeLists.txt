cmake_minimum_required(VERSION 3.10)
project(cnn_cuda LANGUAGES CXX CUDA)

# CUDA settings
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_70,code=sm_70)

# C++ settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "kernels/*.cu")

# Library
cuda_add_library(cnn_cuda SHARED ${SOURCES} ${CUDA_SOURCES})
target_link_libraries(cnn_cuda ${CUDA_LIBRARIES})

# Examples
add_subdirectory(examples)

# Tests
enable_testing()
add_subdirectory(tests)
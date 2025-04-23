#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <sstream>

namespace cnn_cuda {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::stringstream ss; \
            ss << "CUDA error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(ss.str()); \
        } \
    }

// Get CUDA device properties
inline void print_device_info(int device = -1) {
    int current_device;
    if (device < 0) {
        CUDA_CHECK(cudaGetDevice(&current_device));
    } else {
        current_device = device;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, current_device));
    
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "Clock rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
}

// Synchronize CUDA device and check for errors
inline void cuda_sync_check() {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cnn_cuda
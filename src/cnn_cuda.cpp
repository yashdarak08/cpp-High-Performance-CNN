#include "cnn_cuda.h"
#include "utils/memory_pool.h"
#include <cuda_runtime.h>
#include <iostream>

namespace cnn_cuda {

bool init() {
    // Check for CUDA devices
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Get and set the best device
    int best_device = 0;
    int max_multiprocessors = 0;
    
    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        
        if (prop.multiProcessorCount > max_multiprocessors) {
            max_multiprocessors = prop.multiProcessorCount;
            best_device = device;
        }
    }
    
    cudaSetDevice(best_device);
    
    // Initialize memory pool
    MemoryPool::instance();
    
    return true;
}


bool is_cuda_supported() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
}

void shutdown() {
    // Clean up memory pool
    MemoryPool::instance().clear();
    
    // Reset CUDA device
    cudaDeviceReset();
}

} // namespace cnn_cuda
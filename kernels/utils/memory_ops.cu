#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../../include/tensor.h"

namespace cnn_cuda {

// Kernel for initializing memory to zeros
__global__ void zero_memory_kernel(
    float* data,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = 0.0f;
    }
}

// Kernel for initializing memory to a constant value
__global__ void fill_memory_kernel(
    float* data,
    float value,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = value;
    }
}

// Kernel for copying memory between two buffers
__global__ void copy_memory_kernel(
    const float* src,
    float* dst,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

// Host function for zeroing memory
void launch_zero_memory(
    float* data,
    int size,
    cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    zero_memory_kernel<<<grid_size, block_size, 0, stream>>>(data, size);
}

// Host function for filling memory with a constant value
void launch_fill_memory(
    float* data,
    float value,
    int size,
    cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    fill_memory_kernel<<<grid_size, block_size, 0, stream>>>(data, value, size);
}

// Host function for copying memory
void launch_copy_memory(
    const float* src,
    float* dst,
    int size,
    cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    copy_memory_kernel<<<grid_size, block_size, 0, stream>>>(src, dst, size);
}

} // namespace cnn_cuda
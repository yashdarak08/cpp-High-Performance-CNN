#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/tensor.h"

namespace cnn_cuda {

// ReLU activation kernel
__global__ void relu_kernel(
    const float* input,
    float* output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Sigmoid activation kernel
__global__ void sigmoid_kernel(
    const float* input,
    float* output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

// Tanh activation kernel
__global__ void tanh_kernel(
    const float* input,
    float* output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

// Leaky ReLU activation kernel
__global__ void leaky_relu_kernel(
    const float* input,
    float* output,
    float alpha,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float val = input[idx];
        output[idx] = val > 0.0f ? val : alpha * val;
    }
}

// Host function for launching ReLU activation
void launch_relu(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    relu_kernel<<<grid_size, block_size, 0, stream>>>(input, output, size);
}

// Host function for launching sigmoid activation
void launch_sigmoid(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    sigmoid_kernel<<<grid_size, block_size, 0, stream>>>(input, output, size);
}

// Host function for launching tanh activation
void launch_tanh(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    tanh_kernel<<<grid_size, block_size, 0, stream>>>(input, output, size);
}

// Host function for launching leaky ReLU activation
void launch_leaky_relu(
    const float* input,
    float* output,
    float alpha,
    int size,
    cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    leaky_relu_kernel<<<grid_size, block_size, 0, stream>>>(input, output, alpha, size);
}

} // namespace cnn_cuda
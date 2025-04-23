#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/tensor.h"

namespace cnn_cuda {

// CUDA kernel for tensor addition
__global__ void tensor_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

// CUDA kernel for tensor multiplication (element-wise)
__global__ void tensor_mul_kernel(
    const float* a,
    const float* b,
    float* c,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

// CUDA kernel for tensor ReLU
__global__ void tensor_relu_kernel(
    const float* input,
    float* output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Host function for tensor addition
void tensor_add(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Tensor shapes must match for addition");
    }
    
    if (out.shape() != a.shape()) {
        out = Tensor(a.shape());
    }
    
    int size = a.size();
    
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    tensor_add_kernel<<<grid_size, block_size>>>(
        a.data(), b.data(), out.data(), size
    );
}

// Host function for tensor multiplication
void tensor_mul(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Tensor shapes must match for multiplication");
    }
    
    if (out.shape() != a.shape()) {
        out = Tensor(a.shape());
    }
    
    int size = a.size();
    
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    tensor_mul_kernel<<<grid_size, block_size>>>(
        a.data(), b.data(), out.data(), size
    );
}

// Host function for tensor ReLU
void tensor_relu(const Tensor& input, Tensor& output) {
    if (output.shape() != input.shape()) {
        output = Tensor(input.shape());
    }
    
    int size = input.size();
    
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    tensor_relu_kernel<<<grid_size, block_size>>>(
        input.data(), output.data(), size
    );
}

} // namespace cnn_cuda
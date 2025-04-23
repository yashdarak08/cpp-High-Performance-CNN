#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/tensor.h"

namespace cnn_cuda {

// Batch Normalization inference kernel
__global__ void batch_norm_inference_kernel(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    int batch_size,
    int channels,
    int height,
    int width,
    float epsilon)
{
    int n = blockIdx.x;
    int c = blockIdx.y;
    int h = blockIdx.z / width;
    int w = blockIdx.z % width;
    
    if (n >= batch_size || c >= channels || h >= height || w >= width)
        return;
    
    int index = ((n * channels + c) * height + h) * width + w;
    
    // Get the scaling factor for this channel
    float scale = gamma[c] / sqrt(running_var[c] + epsilon);
    float bias = beta[c] - running_mean[c] * scale;
    
    // Normalize
    output[index] = input[index] * scale + bias;
}

// Flattened version for larger inputs
__global__ void flattened_batch_norm_kernel(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    int batch_size,
    int channels,
    int height,
    int width,
    float epsilon)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx >= total_elements)
        return;
    
    // Calculate n, c, h, w from flattened index
    int n = idx / (channels * height * width);
    int c = (idx / (height * width)) % channels;
    int h = (idx / width) % height;
    int w = idx % width;
    
    // Get the scaling factor for this channel
    float scale = gamma[c] / sqrt(running_var[c] + epsilon);
    float bias = beta[c] - running_mean[c] * scale;
    
    // Normalize
    output[idx] = input[idx] * scale + bias;
}

// Host function for launching batch normalization
void launch_batch_norm(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    int batch_size,
    int channels,
    int height,
    int width,
    float epsilon,
    cudaStream_t stream)
{
    // Calculate dimensions
    int hw_size = height * width;
    
    // Check if grid dimensions would be within limits
    if (batch_size > 65535 || channels > 65535 || hw_size > 65535) {
        // If grid is too large, use flattened kernel
        int total_elements = batch_size * channels * hw_size;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;
        
        flattened_batch_norm_kernel<<<grid_size, block_size, 0, stream>>>(
            input, output, gamma, beta, running_mean, running_var,
            batch_size, channels, height, width, epsilon);
    } else {
        // Launch regular kernel
        dim3 block(1, 1, 1);
        dim3 grid(batch_size, channels, hw_size);
        
        batch_norm_inference_kernel<<<grid, block, 0, stream>>>(
            input, output, gamma, beta, running_mean, running_var,
            batch_size, channels, height, width, epsilon);
    }
}

} // namespace cnn_cuda
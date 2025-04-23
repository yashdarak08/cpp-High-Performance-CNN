#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/tensor.h"
#include <float.h>

namespace cnn_cuda {

// CUDA kernel for max pooling
__global__ void max_pool_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride)
{
    int n = blockIdx.x;
    int c = blockIdx.y;
    int h = blockIdx.z / out_width;
    int w = blockIdx.z % out_width;
    
    if (n >= batch_size || c >= channels || h >= out_height || w >= out_width)
        return;
    
    // Input offset for this batch and channel
    int in_offset = ((n * channels + c) * in_height * in_width);
    
    // Output position
    int out_pos = ((n * channels + c) * out_height + h) * out_width + w;
    
    // Compute pooling
    float max_val = -FLT_MAX;
    
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int in_h = h * stride + kh;
            int in_w = w * stride + kw;
            
            if (in_h < in_height && in_w < in_width) {
                float val = input[in_offset + in_h * in_width + in_w];
                max_val = fmaxf(max_val, val);
            }
        }
    }
    
    output[out_pos] = max_val;
}

// CUDA kernel for average pooling
__global__ void avg_pool_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride)
{
    int n = blockIdx.x;
    int c = blockIdx.y;
    int h = blockIdx.z / out_width;
    int w = blockIdx.z % out_width;
    
    if (n >= batch_size || c >= channels || h >= out_height || w >= out_width)
        return;
    
    // Input offset for this batch and channel
    int in_offset = ((n * channels + c) * in_height * in_width);
    
    // Output position
    int out_pos = ((n * channels + c) * out_height + h) * out_width + w;
    
    // Compute pooling
    float sum = 0.0f;
    int count = 0;
    
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int in_h = h * stride + kh;
            int in_w = w * stride + kw;
            
            if (in_h < in_height && in_w < in_width) {
                sum += input[in_offset + in_h * in_width + in_w];
                count++;
            }
        }
    }
    
    output[out_pos] = count > 0 ? sum / count : 0.0f;
}

// Host function for launching max pooling
void launch_max_pool(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    cudaStream_t stream)
{
    // Calculate output dimensions
    int out_height = (in_height - kernel_size) / stride + 1;
    int out_width = (in_width - kernel_size) / stride + 1;
    
    // Launch kernel
    dim3 block(1, 1, 1);  // Each thread handles one output element
    dim3 grid(batch_size, channels, out_height * out_width);
    
    // Check if grid dimensions are within limits
    if (grid.x > 65535 || grid.y > 65535 || grid.z > 65535) {
        // If grid is too large, flatten it and use more threads per block
        int total_elements = batch_size * channels * out_height * out_width;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;
        
        // Launch a flattened kernel instead (not implemented here)
        // flatten_max_pool_kernel<<<grid_size, block_size, 0, stream>>>(
        //    input, output, batch_size, channels, in_height, in_width,
        //    out_height, out_width, kernel_size, stride);
    } else {
        // Launch the regular kernel
        max_pool_kernel<<<grid, block, 0, stream>>>(
            input, output, batch_size, channels, in_height, in_width,
            out_height, out_width, kernel_size, stride);
    }
}

// Host function for launching average pooling
void launch_avg_pool(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    cudaStream_t stream)
{
    // Calculate output dimensions
    int out_height = (in_height - kernel_size) / stride + 1;
    int out_width = (in_width - kernel_size) / stride + 1;
    
    // Launch kernel
    dim3 block(1, 1, 1);  // Each thread handles one output element
    dim3 grid(batch_size, channels, out_height * out_width);
    
    // Check if grid dimensions are within limits
    if (grid.x > 65535 || grid.y > 65535 || grid.z > 65535) {
        // If grid is too large, flatten it and use more threads per block
        int total_elements = batch_size * channels * out_height * out_width;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;
        
        // Launch a flattened kernel instead (not implemented here)
        // flatten_avg_pool_kernel<<<grid_size, block_size, 0, stream>>>(
        //    input, output, batch_size, channels, in_height, in_width,
        //    out_height, out_width, kernel_size, stride);
    } else {
        // Launch the regular kernel
        avg_pool_kernel<<<grid, block, 0, stream>>>(
            input, output, batch_size, channels, in_height, in_width,
            out_height, out_width, kernel_size, stride);
    }
}

// CUDA kernel for global average pooling
__global__ void global_avg_pool_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width)
{
    int n = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (n >= batch_size || c >= channels)
        return;
    
    // Input offset for this batch and channel
    int in_offset = (n * channels + c) * height * width;
    
    // Output position
    int out_pos = n * channels + c;
    
    // Compute global average
    float sum = 0.0f;
    int count = height * width;
    
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            sum += input[in_offset + h * width + w];
        }
    }
    
    output[out_pos] = sum / count;
}

// Host function for launching global average pooling
void launch_global_avg_pool(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    cudaStream_t stream)
{
    // Launch kernel
    int block_size = 256;
    dim3 grid((channels + block_size - 1) / block_size, batch_size);
    
    global_avg_pool_kernel<<<grid, block_size, 0, stream>>>(
        input, output, batch_size, channels, height, width);
}

// CUDA kernel for flattened max pooling (handling large inputs)
__global__ void flatten_max_pool_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_height * out_width;
    
    if (idx >= total_elements)
        return;
    
    // Calculate n, c, h, w from flattened index
    int n = idx / (channels * out_height * out_width);
    int c = (idx / (out_height * out_width)) % channels;
    int h = (idx / out_width) % out_height;
    int w = idx % out_width;
    
    // Input offset for this batch and channel
    int in_offset = ((n * channels + c) * in_height * in_width);
    
    // Compute pooling
    float max_val = -FLT_MAX;
    
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int in_h = h * stride + kh;
            int in_w = w * stride + kw;
            
            if (in_h < in_height && in_w < in_width) {
                float val = input[in_offset + in_h * in_width + in_w];
                max_val = fmaxf(max_val, val);
            }
        }
    }
    
    output[idx] = max_val;
}

// CUDA kernel for flattened average pooling (handling large inputs)
__global__ void flatten_avg_pool_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_height * out_width;
    
    if (idx >= total_elements)
        return;
    
    // Calculate n, c, h, w from flattened index
    int n = idx / (channels * out_height * out_width);
    int c = (idx / (out_height * out_width)) % channels;
    int h = (idx / out_width) % out_height;
    int w = idx % out_width;
    
    // Input offset for this batch and channel
    int in_offset = ((n * channels + c) * in_height * in_width);
    
    // Compute pooling
    float sum = 0.0f;
    int count = 0;
    
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int in_h = h * stride + kh;
            int in_w = w * stride + kw;
            
            if (in_h < in_height && in_w < in_width) {
                sum += input[in_offset + in_h * in_width + in_w];
                count++;
            }
        }
    }
    
    output[idx] = count > 0 ? sum / count : 0.0f;
}

// Update the launch functions to use the flattened kernels when needed
void launch_max_pool_flattened(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    cudaStream_t stream)
{
    // Calculate output dimensions
    int out_height = (in_height - kernel_size) / stride + 1;
    int out_width = (in_width - kernel_size) / stride + 1;
    
    // Calculate total number of output elements
    int total_elements = batch_size * channels * out_height * out_width;
    
    // Launch flattened kernel
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    flatten_max_pool_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, batch_size, channels, in_height, in_width,
        out_height, out_width, kernel_size, stride);
}

void launch_avg_pool_flattened(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    cudaStream_t stream)
{
    // Calculate output dimensions
    int out_height = (in_height - kernel_size) / stride + 1;
    int out_width = (in_width - kernel_size) / stride + 1;
    
    // Calculate total number of output elements
    int total_elements = batch_size * channels * out_height * out_width;
    
    // Launch flattened kernel
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    flatten_avg_pool_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, batch_size, channels, in_height, in_width,
        out_height, out_width, kernel_size, stride);
}

} // namespace cnn_cuda
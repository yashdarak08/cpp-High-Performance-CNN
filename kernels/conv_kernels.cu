#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/tensor.h"
#include <stdio.h>

namespace cnn_cuda {

// Direct convolution kernel for small kernels (3x3, 5x5)
__global__ void direct_conv2d_kernel(
    const float* input, 
    const float* weights,
    float* output,
    int batch_size, 
    int in_channels,
    int out_channels,
    int in_height, 
    int in_width,
    int kernel_size,
    int out_height, 
    int out_width,
    int stride, 
    int padding) 
{
    int n = blockIdx.x;                          // batch dimension
    int m = blockIdx.y;                          // output channel dimension
    int h = (blockIdx.z / out_width) * stride;  // height position
    int w = (blockIdx.z % out_width) * stride;  // width position
    
    // Output position
    int out_pos = ((n * out_channels + m) * out_height + (h/stride)) * out_width + (w/stride);
    
    // Accumulate sum for the convolution
    float sum = 0.0f;
    
    // Compute convolution for the (h,w) output element
    for (int c = 0; c < in_channels; c++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in = h - padding + kh;
                int w_in = w - padding + kw;
                
                // Check if input position is valid
                if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                    int in_pos = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                    int weight_pos = ((m * in_channels + c) * kernel_size + kh) * kernel_size + kw;
                    
                    sum += input[in_pos] * weights[weight_pos];
                }
            }
        }
    }
    
    output[out_pos] = sum;
}

// CUDA kernel to perform matrix multiplication for im2col+GEMM approach
__global__ void im2col_gemm_kernel(
    const float* input_col,
    const float* weights,
    float* output,
    int batch_size,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_size,
    int in_channels) 
{
    int batch = blockIdx.y;
    int out_c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_c >= out_channels) return;
    
    // Number of elements in output feature map
    int out_size = out_height * out_width;
    
    // Number of elements in each filter
    int filter_size = kernel_size * kernel_size * in_channels;
    
    // Output position for this batch and output channel
    int out_offset = batch * out_channels * out_size + out_c * out_size;
    
    // Input matrix position for this batch
    int in_offset = batch * filter_size * out_size;
    
    // Weight matrix position for this output channel
    int w_offset = out_c * filter_size;
    
    // Loop through all spatial positions in the output
    for (int out_pos = 0; out_pos < out_size; ++out_pos) {
        float sum = 0.0f;
        
        // Compute dot product between weight row and input column
        for (int i = 0; i < filter_size; ++i) {
            sum += weights[w_offset + i] * input_col[in_offset + out_pos * filter_size + i];
        }
        
        output[out_offset + out_pos] = sum;
    }
}

// Helper function to transform input data into columns (im2col)
__global__ void im2col_kernel(
    const float* input,
    float* output_col,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int out_height,
    int out_width,
    int stride,
    int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = out_height * out_width * batch_size;
    
    if (idx >= total_threads) return;
    
    // Determine which output position and batch we're working on
    int out_pos = idx % (out_height * out_width);
    int batch = idx / (out_height * out_width);
    
    // Calculate output (x,y) coordinates
    int out_y = out_pos / out_width;
    int out_x = out_pos % out_width;
    
    // Calculate top-left input position for this filter application
    int in_y = out_y * stride - padding;
    int in_x = out_x * stride - padding;
    
    // For each input channel and each position in the kernel
    for (int c = 0; c < in_channels; ++c) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int col_idx = ((batch * out_height * out_width + out_pos) * 
                              (in_channels * kernel_size * kernel_size) +
                               c * kernel_size * kernel_size +
                               ky * kernel_size + kx);
                
                // Calculate input position
                int y = in_y + ky;
                int x = in_x + kx;
                
                // Check if the input position is valid
                if (y >= 0 && y < in_height && x >= 0 && x < in_width) {
                    int in_idx = ((batch * in_channels + c) * in_height + y) * in_width + x;
                    output_col[col_idx] = input[in_idx];
                } else {
                    output_col[col_idx] = 0.0f;  // Zero padding
                }
            }
        }
    }
}

// Host function for launching direct convolution
void launch_direct_conv2d(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    cudaStream_t stream)
{
    // Calculate output dimensions
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    // Launch direct convolution kernel
    dim3 grid(batch_size, out_channels, out_height * out_width);
    direct_conv2d_kernel<<<grid, 1, 0, stream>>>(
        input, weights, output,
        batch_size, in_channels, out_channels,
        in_height, in_width, kernel_size,
        out_height, out_width, stride, padding
    );
    
    // Kernel to add bias can be launched separately
    // (omitted for brevity)
}

// Host function for launching im2col + GEMM convolution
void launch_im2col_gemm_conv2d(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    cudaStream_t stream)
{
    // Calculate output dimensions
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    // Calculate sizes
    int out_size = out_height * out_width;
    int filter_size = kernel_size * kernel_size * in_channels;
    
    // Allocate memory for im2col output
    float* input_col;
    cudaMalloc(&input_col, sizeof(float) * batch_size * out_size * filter_size);
    
    // Launch im2col kernel
    int block_size = 256;
    int grid_size = (batch_size * out_height * out_width + block_size - 1) / block_size;
    
    im2col_kernel<<<grid_size, block_size, 0, stream>>>(
        input, input_col,
        batch_size, in_channels, in_height, in_width,
        kernel_size, out_height, out_width, stride, padding
    );
    
    // Launch GEMM kernel
    dim3 gemm_block(32, 1);
    dim3 gemm_grid((out_channels + gemm_block.x - 1) / gemm_block.x, batch_size);
    
    im2col_gemm_kernel<<<gemm_grid, gemm_block, 0, stream>>>(
        input_col, weights, output,
        batch_size, out_channels, out_height, out_width,
        kernel_size, in_channels
    );
    
    // Clean up
    cudaFree(input_col);
    
    // Kernel to add bias can be launched separately
    // (omitted for brevity)
}

} // namespace cnn_cuda
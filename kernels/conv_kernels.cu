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
    
}

// Winograd F(2x2, 3x3) transformation matrices
// These matrices are used to transform the input and filter
// B^T d B for data, G g G^T for filter, A^T m A for output
__device__ __constant__ float d_winograd_B[4][4] = {
    {1.0f, 0.0f, -1.0f, 0.0f},
    {0.0f, 1.0f, 1.0f, 0.0f},
    {0.0f, -1.0f, 1.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, -1.0f}
};

__device__ __constant__ float d_winograd_G[4][3] = {
    {1.0f, 0.0f, 0.0f},
    {0.5f, 0.5f, 0.5f},
    {0.5f, -0.5f, 0.5f},
    {0.0f, 0.0f, 1.0f}
};

__device__ __constant__ float d_winograd_A[2][4] = {
    {1.0f, 1.0f, 1.0f, 0.0f},
    {0.0f, 1.0f, -1.0f, 1.0f}
};

// CUDA kernel for Winograd input tile transformation
__global__ void winograd_input_transform_kernel(
    const float* input,
    float* input_transformed,
    int batch_size,
    int channels,
    int height,
    int width,
    int tile_h,
    int tile_w,
    int padding)
{
    int n = blockIdx.x;
    int c = blockIdx.y;
    int tile_idx = blockIdx.z;
    
    int tile_y = tile_idx / tile_w;
    int tile_x = tile_idx % tile_w;
    
    // Calculate input tile offset
    int in_offset = ((n * channels + c) * height * width);
    
    // Calculate tile top-left position with padding consideration
    int start_y = tile_y * 2 - padding;
    int start_x = tile_x * 2 - padding;
    
    // Prepare a 4x4 tile of data for transformation
    float tile_data[4][4] = {0};
    
    // Load tile data with zero padding for out-of-bounds
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            int in_y = start_y + y;
            int in_x = start_x + x;
            
            if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                tile_data[y][x] = input[in_offset + in_y * width + in_x];
            }
        }
    }
    
    // Apply winograd transformation B^T * d * B
    float transformed[4][4] = {0};
    
    // B^T * d
    float temp[4][4] = {0};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                temp[i][j] += d_winograd_B[k][i] * tile_data[k][j];
            }
        }
    }
    
    // (B^T * d) * B
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                transformed[i][j] += temp[i][k] * d_winograd_B[k][j];
            }
        }
    }
    
    // Write to output with appropriate offset
    int out_offset = (((n * channels + c) * tile_h * tile_w) + tile_y * tile_w + tile_x) * 16;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            input_transformed[out_offset + i * 4 + j] = transformed[i][j];
        }
    }
}

// CUDA kernel for Winograd filter transformation
__global__ void winograd_filter_transform_kernel(
    const float* filter,
    float* filter_transformed,
    int out_channels,
    int in_channels)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (m >= out_channels || c >= in_channels) {
        return;
    }
    
    // Load the 3x3 filter
    float filter_data[3][3];
    int filter_offset = (m * in_channels + c) * 9;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            filter_data[i][j] = filter[filter_offset + i * 3 + j];
        }
    }
    
    // Apply G * g * G^T transformation
    float transformed[4][4] = {0};
    
    // G * g
    float temp[4][3] = {0};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                temp[i][j] += d_winograd_G[i][k] * filter_data[k][j];
            }
        }
    }
    
    // (G * g) * G^T
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 3; ++k) {
                transformed[i][j] += temp[i][k] * d_winograd_G[j][k];
            }
        }
    }
    
    // Write to output with appropriate offset
    int out_offset = (m * in_channels + c) * 16;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            filter_transformed[out_offset + i * 4 + j] = transformed[i][j];
        }
    }
}

// CUDA kernel for element-wise multiplication of transformed input and filter
__global__ void winograd_element_wise_mul_kernel(
    const float* input_transformed,
    const float* filter_transformed,
    float* output_transformed,
    int batch_size,
    int out_channels,
    int in_channels,
    int tile_h,
    int tile_w)
{
    int n = blockIdx.x;
    int m = blockIdx.y;
    int tile_idx = blockIdx.z;
    
    // For each input channel, perform the element-wise multiplication
    // and accumulate the results
    float sum[16] = {0};
    
    for (int c = 0; c < in_channels; ++c) {
        int in_offset = (((n * in_channels + c) * tile_h * tile_w) + tile_idx) * 16;
        int filter_offset = (m * in_channels + c) * 16;
        
        for (int i = 0; i < 16; ++i) {
            sum[i] += input_transformed[in_offset + i] * filter_transformed[filter_offset + i];
        }
    }
    
    // Write to output with appropriate offset
    int out_offset = (((n * out_channels + m) * tile_h * tile_w) + tile_idx) * 16;
    for (int i = 0; i < 16; ++i) {
        output_transformed[out_offset + i] = sum[i];
    }
}

// CUDA kernel for Winograd output transformation
__global__ void winograd_output_transform_kernel(
    const float* output_transformed,
    float* output,
    const float* bias,
    int batch_size,
    int out_channels,
    int out_height,
    int out_width,
    int tile_h,
    int tile_w)
{
    int n = blockIdx.x;
    int m = blockIdx.y;
    int tile_idx = blockIdx.z;
    
    int tile_y = tile_idx / tile_w;
    int tile_x = tile_idx % tile_w;
    
    // Load the transformed output
    float transformed[4][4] = {0};
    int in_offset = (((n * out_channels + m) * tile_h * tile_w) + tile_idx) * 16;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            transformed[i][j] = output_transformed[in_offset + i * 4 + j];
        }
    }
    
    // Apply A^T * m * A transformation
    float result[2][2] = {0};
    
    // A^T * m
    float temp[2][4] = {0};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                temp[i][j] += d_winograd_A[i][k] * transformed[k][j];
            }
        }
    }
    
    // (A^T * m) * A
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 4; ++k) {
                result[i][j] += temp[i][k] * d_winograd_A[j][k];
            }
        }
    }
    
    // Write to output
    int out_y = tile_y * 2;
    int out_x = tile_x * 2;
    int out_offset = ((n * out_channels + m) * out_height + out_y) * out_width + out_x;
    
    // Add bias
    float bias_value = bias[m];
    
    // Write the 2x2 output tile if within bounds
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            if (out_y + y < out_height && out_x + x < out_width) {
                output[out_offset + y * out_width + x] = result[y][x] + bias_value;
            }
        }
    }
}

// Host function for launching Winograd convolution
void launch_winograd_conv2d(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int padding,
    cudaStream_t stream)
{
    // Calculate output dimensions
    int out_height = in_height + 2 * padding - 2;
    int out_width = in_width + 2 * padding - 2;
    
    // Calculate number of tiles
    int tile_h = (out_height + 1) / 2;
    int tile_w = (out_width + 1) / 2;
    
    // Allocate memory for transformed data
    float* input_transformed;
    float* filter_transformed;
    float* output_transformed;
    
    size_t input_transformed_size = batch_size * in_channels * tile_h * tile_w * 16 * sizeof(float);
    size_t filter_transformed_size = out_channels * in_channels * 16 * sizeof(float);
    size_t output_transformed_size = batch_size * out_channels * tile_h * tile_w * 16 * sizeof(float);
    
    cudaMalloc(&input_transformed, input_transformed_size);
    cudaMalloc(&filter_transformed, filter_transformed_size);
    cudaMalloc(&output_transformed, output_transformed_size);
    
    // 1. Transform input
    dim3 input_block(1, 1, 1);
    dim3 input_grid(batch_size, in_channels, tile_h * tile_w);
    
    winograd_input_transform_kernel<<<input_grid, input_block, 0, stream>>>(
        input, input_transformed, batch_size, in_channels, in_height, in_width, 
        tile_h, tile_w, padding
    );
    
    // 2. Transform filter
    dim3 filter_block(16, 16);
    dim3 filter_grid((out_channels + 15) / 16, (in_channels + 15) / 16);
    
    winograd_filter_transform_kernel<<<filter_grid, filter_block, 0, stream>>>(
        weights, filter_transformed, out_channels, in_channels
    );
    
    // 3. Element-wise multiply and accumulate
    dim3 mul_block(1, 1, 1);
    dim3 mul_grid(batch_size, out_channels, tile_h * tile_w);
    
    winograd_element_wise_mul_kernel<<<mul_grid, mul_block, 0, stream>>>(
        input_transformed, filter_transformed, output_transformed,
        batch_size, out_channels, in_channels, tile_h, tile_w
    );
    
    // 4. Transform output
    dim3 output_block(1, 1, 1);
    dim3 output_grid(batch_size, out_channels, tile_h * tile_w);
    
    winograd_output_transform_kernel<<<output_grid, output_block, 0, stream>>>(
        output_transformed, output, bias, batch_size, out_channels,
        out_height, out_width, tile_h, tile_w
    );
    
    // Clean up
    cudaFree(input_transformed);
    cudaFree(filter_transformed);
    cudaFree(output_transformed);
}

} // namespace cnn_cuda
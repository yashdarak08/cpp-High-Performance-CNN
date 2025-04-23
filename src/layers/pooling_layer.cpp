#include "layers/pooling_layer.h"
#include <stdexcept>

namespace cnn_cuda {

// Forward declarations of CUDA kernel launcher functions
void launch_max_pool(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    cudaStream_t stream);

void launch_avg_pool(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    cudaStream_t stream);

MaxPoolingLayer::MaxPoolingLayer(int kernel_size, int stride)
    : kernel_size_(kernel_size), stride_(stride) {}

Tensor MaxPoolingLayer::forward(const Tensor& input) {
    // Input shape: [batch_size, channels, in_height, in_width]
    const std::vector<int>& in_shape = input.shape();
    
    if (in_shape.size() != 4) {
        throw std::runtime_error("Input must be 4D tensor [batch_size, channels, height, width]");
    }
    
    int batch_size = in_shape[0];
    int channels = in_shape[1];
    int in_height = in_shape[2];
    int in_width = in_shape[3];
    
    // Calculate output dimensions
    int out_height = (in_height - kernel_size_) / stride_ + 1;
    int out_width = (in_width - kernel_size_) / stride_ + 1;
    
    // Create output tensor
    Tensor output({batch_size, channels, out_height, out_width});
    
    // Launch CUDA kernel
    launch_max_pool(
        input.data(),
        output.data(),
        batch_size,
        channels,
        in_height,
        in_width,
        kernel_size_,
        stride_,
        0 // Default stream
    );
    
    return output;
}

AvgPoolingLayer::AvgPoolingLayer(int kernel_size, int stride)
    : kernel_size_(kernel_size), stride_(stride) {}

Tensor AvgPoolingLayer::forward(const Tensor& input) {
    // Input shape: [batch_size, channels, in_height, in_width]
    const std::vector<int>& in_shape = input.shape();
    
    if (in_shape.size() != 4) {
        throw std::runtime_error("Input must be 4D tensor [batch_size, channels, height, width]");
    }
    
    int batch_size = in_shape[0];
    int channels = in_shape[1];
    int in_height = in_shape[2];
    int in_width = in_shape[3];
    
    // Calculate output dimensions
    int out_height = (in_height - kernel_size_) / stride_ + 1;
    int out_width = (in_width - kernel_size_) / stride_ + 1;
    
    // Create output tensor
    Tensor output({batch_size, channels, out_height, out_width});
    
    // Launch CUDA kernel
    launch_avg_pool(
        input.data(),
        output.data(),
        batch_size,
        channels,
        in_height,
        in_width,
        kernel_size_,
        stride_,
        0 // Default stream
    );
    
    return output;
}

} // namespace cnn_cuda
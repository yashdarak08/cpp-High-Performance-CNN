#include "layers/conv_layer.h"
#include <random>
#include <algorithm>
#include <cmath>

namespace cnn_cuda {

// Forward declarations of CUDA kernel launcher functions
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
    cudaStream_t stream);

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
    cudaStream_t stream);

ConvLayer::ConvLayer(int in_channels, int out_channels, int kernel_size, 
                     int stride, int padding)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding) {
    
    // Initialize weights and bias tensors
    weights_ = Tensor({out_channels_, in_channels_, kernel_size_, kernel_size_});
    bias_ = Tensor({out_channels_});
}

void ConvLayer::initialize_parameters() {
    // Initialize weights with Kaiming normal initialization
    float std_dev = std::sqrt(2.0f / (in_channels_ * kernel_size_ * kernel_size_));
    
    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    // Generate random weights on CPU first
    std::vector<float> weight_data(weights_.size());
    std::generate(weight_data.begin(), weight_data.end(), [&]() { return dist(gen); });
    
    // Create CPU tensor, then copy to GPU
    Tensor cpu_weights(weights_.shape(), weight_data.data(), DeviceType::CPU);
    weights_.copy_from(cpu_weights);
    
    // Initialize bias to zeros
    std::vector<float> bias_data(bias_.size(), 0.0f);
    Tensor cpu_bias(bias_.shape(), bias_data.data(), DeviceType::CPU);
    bias_.copy_from(cpu_bias);
}

void ConvLayer::load_weights(const float* weights, const float* bias) {
    // Copy weights and bias from host to device
    Tensor cpu_weights(weights_.shape(), weights, DeviceType::CPU);
    weights_.copy_from(cpu_weights);
    
    Tensor cpu_bias(bias_.shape(), bias, DeviceType::CPU);
    bias_.copy_from(cpu_bias);
}

const Tensor& ConvLayer::get_weights() const {
    return weights_;
}

const Tensor& ConvLayer::get_bias() const {
    return bias_;
}

ConvLayer::ConvImplementation ConvLayer::choose_implementation(const Tensor& input) const {
    // Simple heuristic for choosing implementation
    if (kernel_size_ == 3 && in_channels_ >= 16 && out_channels_ >= 16) {
        return ConvImplementation::WINOGRAD;
    } else if (kernel_size_ > 5 || (in_channels_ >= 64 && out_channels_ >= 64)) {
        return ConvImplementation::IM2COL_GEMM;
    } else {
        return ConvImplementation::DIRECT;
    }
}

Tensor ConvLayer::forward(const Tensor& input) {
    // Choose implementation based on input size and kernel size
    ConvImplementation impl = choose_implementation(input);
    
    // Call appropriate implementation
    switch (impl) {
        case ConvImplementation::DIRECT:
            return forward_direct(input);
        case ConvImplementation::IM2COL_GEMM:
            return forward_im2col_gemm(input);
        case ConvImplementation::WINOGRAD:
            return forward_winograd(input);
        default:
            return forward_direct(input);
    }
}

Tensor ConvLayer::forward_direct(const Tensor& input) {
    // Input shape: [batch_size, in_channels, in_height, in_width]
    const std::vector<int>& in_shape = input.shape();
    
    if (in_shape.size() != 4) {
        throw std::runtime_error("Input must be 4D tensor [batch_size, channels, height, width]");
    }
    
    int batch_size = in_shape[0];
    int in_height = in_shape[2];
    int in_width = in_shape[3];
    
    // Calculate output dimensions
    int out_height = (in_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    int out_width = (in_width + 2 * padding_ - kernel_size_) / stride_ + 1;
    
    // Create output tensor
    Tensor output({batch_size, out_channels_, out_height, out_width});
    
    // Launch CUDA kernel
    launch_direct_conv2d(
        input.data(),
        weights_.data(),
        bias_.data(),
        output.data(),
        batch_size,
        in_channels_,
        out_channels_,
        in_height,
        in_width,
        kernel_size_,
        stride_,
        padding_,
        0 // Default stream
    );
    
    return output;
}

Tensor ConvLayer::forward_im2col_gemm(const Tensor& input) {
    // Input shape: [batch_size, in_channels, in_height, in_width]
    const std::vector<int>& in_shape = input.shape();
    
    if (in_shape.size() != 4) {
        throw std::runtime_error("Input must be 4D tensor [batch_size, channels, height, width]");
    }
    
    int batch_size = in_shape[0];
    int in_height = in_shape[2];
    int in_width = in_shape[3];
    
    // Calculate output dimensions
    int out_height = (in_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    int out_width = (in_width + 2 * padding_ - kernel_size_) / stride_ + 1;
    
    // Create output tensor
    Tensor output({batch_size, out_channels_, out_height, out_width});
    
    // Launch CUDA kernel
    launch_im2col_gemm_conv2d(
        input.data(),
        weights_.data(),
        bias_.data(),
        output.data(),
        batch_size,
        in_channels_,
        out_channels_,
        in_height,
        in_width,
        kernel_size_,
        stride_,
        padding_,
        0 // Default stream
    );
    
    return output;
}

Tensor ConvLayer::forward_winograd(const Tensor& input) {
    // Winograd implementation would go here
    // For now, fall back to im2col_gemm
    return forward_im2col_gemm(input);
}

} // namespace cnn_cuda
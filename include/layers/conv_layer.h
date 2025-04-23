#pragma once

#include "tensor.h"
#include <vector>
#include <memory>

namespace cnn_cuda {

class ConvLayer {
public:
    ConvLayer(int in_channels, int out_channels, int kernel_size, 
             int stride = 1, int padding = 0);
    
    // Initialize weights and biases
    void initialize_parameters();
    
    // Forward pass
    Tensor forward(const Tensor& input);
    
    // Load pre-trained weights
    void load_weights(const float* weights, const float* bias);
    
    // Get layer parameters
    const Tensor& get_weights() const;
    const Tensor& get_bias() const;

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
   
    Tensor weights_; // Shape: [out_channels, in_channels, kernel_size, kernel_size]
    Tensor bias_;    // Shape: [out_channels]
    
    // Implementation strategies
    enum class ConvImplementation {
        DIRECT,
        IM2COL_GEMM,
        WINOGRAD
    };
    
    // Choose best implementation based on input size and kernel size
    ConvImplementation choose_implementation(const Tensor& input) const;
    
    // Implementation-specific forward methods
    Tensor forward_direct(const Tensor& input);
    Tensor forward_im2col_gemm(const Tensor& input);
    Tensor forward_winograd(const Tensor& input);
};

} // namespace cnn_cuda
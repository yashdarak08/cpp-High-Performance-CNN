#pragma once

#include "tensor.h"

namespace cnn_cuda {

enum class ActivationType {
    RELU,
    SIGMOID,
    TANH,
    LEAKY_RELU
};

class ActivationLayer {
public:
    ActivationLayer(ActivationType type, float alpha = 0.01f);
    
    // Forward pass
    Tensor forward(const Tensor& input);

private:
    ActivationType type_;
    float alpha_; // For leaky ReLU
};

// Convenience functions for common activations
Tensor relu(const Tensor& input);
Tensor sigmoid(const Tensor& input);
Tensor tanh(const Tensor& input);
Tensor leaky_relu(const Tensor& input, float alpha = 0.01f);

} // namespace cnn_cuda
#include "layers/activation_layer.h"
#include <stdexcept>

namespace cnn_cuda {

// Forward declarations of CUDA kernel launcher functions
void launch_relu(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream);

void launch_sigmoid(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream);

void launch_tanh(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream);

void launch_leaky_relu(
    const float* input,
    float* output,
    float alpha,
    int size,
    cudaStream_t stream);

ActivationLayer::ActivationLayer(ActivationType type, float alpha)
    : type_(type), alpha_(alpha) {}

Tensor ActivationLayer::forward(const Tensor& input) {
    // Create output tensor with same shape as input
    Tensor output(input.shape());
    
    // Get total number of elements
    int size = input.size();
    
    // Launch appropriate CUDA kernel based on activation type
    switch (type_) {
        case ActivationType::RELU:
            launch_relu(input.data(), output.data(), size, 0);
            break;
        case ActivationType::SIGMOID:
            launch_sigmoid(input.data(), output.data(), size, 0);
            break;
        case ActivationType::TANH:
            launch_tanh(input.data(), output.data(), size, 0);
            break;
        case ActivationType::LEAKY_RELU:
            launch_leaky_relu(input.data(), output.data(), alpha_, size, 0);
            break;
        default:
            throw std::runtime_error("Unknown activation type");
    }
    
    return output;
}

// Convenience functions for common activations
Tensor relu(const Tensor& input) {
    ActivationLayer layer(ActivationType::RELU);
    return layer.forward(input);
}

Tensor sigmoid(const Tensor& input) {
    ActivationLayer layer(ActivationType::SIGMOID);
    return layer.forward(input);
}

Tensor tanh(const Tensor& input) {
    ActivationLayer layer(ActivationType::TANH);
    return layer.forward(input);
}

Tensor leaky_relu(const Tensor& input, float alpha) {
    ActivationLayer layer(ActivationType::LEAKY_RELU, alpha);
    return layer.forward(input);
}

} // namespace cnn_cuda
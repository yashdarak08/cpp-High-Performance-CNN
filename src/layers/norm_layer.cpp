#include "layers/norm_layer.h"
#include <cuda_runtime.h>

namespace cnn_cuda {

// Forward declaration of CUDA kernel launcher function
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
    cudaStream_t stream);

BatchNormLayer::BatchNormLayer(int num_features, float eps, float momentum)
    : num_features_(num_features), eps_(eps), momentum_(momentum) {
    
    // Initialize parameters
    gamma_ = Tensor({num_features_});
    beta_ = Tensor({num_features_});
    running_mean_ = Tensor({num_features_});
    running_var_ = Tensor({num_features_});
    
    // Initialize gamma to ones and beta to zeros
    std::vector<float> gamma_data(num_features_, 1.0f);
    std::vector<float> beta_data(num_features_, 0.0f);
    std::vector<float> mean_data(num_features_, 0.0f);
    std::vector<float> var_data(num_features_, 1.0f);
    
    Tensor cpu_gamma({num_features_}, gamma_data.data(), DeviceType::CPU);
    Tensor cpu_beta({num_features_}, beta_data.data(), DeviceType::CPU);
    Tensor cpu_mean({num_features_}, mean_data.data(), DeviceType::CPU);
    Tensor cpu_var({num_features_}, var_data.data(), DeviceType::CPU);
    
    gamma_.copy_from(cpu_gamma);
    beta_.copy_from(cpu_beta);
    running_mean_.copy_from(cpu_mean);
    running_var_.copy_from(cpu_var);
}

Tensor BatchNormLayer::forward(const Tensor& input) {
    // Input shape: [batch_size, channels, height, width]
    const std::vector<int>& in_shape = input.shape();
    
    if (in_shape.size() != 4) {
        throw std::runtime_error("Input must be 4D tensor [batch_size, channels, height, width]");
    }
    
    if (in_shape[1] != num_features_) {
        throw std::runtime_error("Input channels must match num_features");
    }
    
    int batch_size = in_shape[0];
    int channels = in_shape[1];
    int height = in_shape[2];
    int width = in_shape[3];
    
    // Create output tensor with same shape as input
    Tensor output(in_shape);
    
    // Launch CUDA kernel
    launch_batch_norm(
        input.data(),
        output.data(),
        gamma_.data(),
        beta_.data(),
        running_mean_.data(),
        running_var_.data(),
        batch_size,
        channels,
        height,
        width,
        eps_,
        0 // Default stream
    );
    
    return output;
}

void BatchNormLayer::load_parameters(
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var) {
    
    // Copy parameters from host to device
    Tensor cpu_gamma({num_features_}, gamma, DeviceType::CPU);
    Tensor cpu_beta({num_features_}, beta, DeviceType::CPU);
    Tensor cpu_mean({num_features_}, running_mean, DeviceType::CPU);
    Tensor cpu_var({num_features_}, running_var, DeviceType::CPU);
    
    gamma_.copy_from(cpu_gamma);
    beta_.copy_from(cpu_beta);
    running_mean_.copy_from(cpu_mean);
    running_var_.copy_from(cpu_var);
}

const Tensor& BatchNormLayer::get_gamma() const {
    return gamma_;
}

const Tensor& BatchNormLayer::get_beta() const {
    return beta_;
}

const Tensor& BatchNormLayer::get_running_mean() const {
    return running_mean_;
}

const Tensor& BatchNormLayer::get_running_var() const {
    return running_var_;
}

} // namespace cnn_cuda
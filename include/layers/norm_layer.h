#pragma once

#include "tensor.h"
#include <vector>

namespace cnn_cuda {

class BatchNormLayer {
public:
    BatchNormLayer(int num_features, float eps = 1e-5f, float momentum = 0.1f);
    
    // Forward pass (inference mode)
    Tensor forward(const Tensor& input);
    
    // Load pre-trained parameters
    void load_parameters(const float* gamma, const float* beta, 
                         const float* running_mean, const float* running_var);
    
    // Get parameters
    const Tensor& get_gamma() const;
    const Tensor& get_beta() const;
    const Tensor& get_running_mean() const;
    const Tensor& get_running_var() const;

private:
    int num_features_;
    float eps_;
    float momentum_;
    
    Tensor gamma_;
    Tensor beta_;
    Tensor running_mean_;
    Tensor running_var_;
};

} // namespace cnn_cuda
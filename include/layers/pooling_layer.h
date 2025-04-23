#pragma once

#include "tensor.h"
#include <vector>

namespace cnn_cuda {

enum class PoolingType {
    MAX,
    AVERAGE
};

class MaxPoolingLayer {
public:
    MaxPoolingLayer(int kernel_size, int stride);
    
    // Forward pass
    Tensor forward(const Tensor& input);

private:
    int kernel_size_;
    int stride_;
};

class AvgPoolingLayer {
public:
    AvgPoolingLayer(int kernel_size, int stride);
    
    // Forward pass
    Tensor forward(const Tensor& input);

private:
    int kernel_size_;
    int stride_;
};

} // namespace cnn_cuda
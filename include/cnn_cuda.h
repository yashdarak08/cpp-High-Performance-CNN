#pragma once

// Main header file that includes all components

#include "tensor.h"
#include "layers/conv_layer.h"
#include "layers/pooling_layer.h"
#include "layers/activation_layer.h"
#include "layers/norm_layer.h"
#include "utils/memory_pool.h"
#include "utils/image_utils.h"

namespace cnn_cuda {

// Library version
constexpr const char* VERSION = "0.1.0";

// Initialize CUDA devices and memory pool
bool init();

// Check if CUDA is supported
bool is_cuda_supported();

// Clean up resources
void shutdown();

} // namespace cnn_cuda
#include "layers/conv_layer.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

using namespace cnn_cuda;

// Test convolution layer initialization
TEST(ConvTest, Initialization) {
    ConvLayer conv(3, 16, 3, 1, 1);
    
    // Check weights shape
    const Tensor& weights = conv.get_weights();
    const auto& w_shape = weights.shape();
    
    EXPECT_EQ(w_shape.size(), 4);
    EXPECT_EQ(w_shape[0], 16);  // out_channels
    EXPECT_EQ(w_shape[1], 3);   // in_channels
    EXPECT_EQ(w_shape[2], 3);   // kernel_height
    EXPECT_EQ(w_shape[3], 3);   // kernel_width
    
    // Check bias shape
    const Tensor& bias = conv.get_bias();
    const auto& b_shape = bias.shape();
    
    EXPECT_EQ(b_shape.size(), 1);
    EXPECT_EQ(b_shape[0], 16);  // out_channels
}

// Test forward pass with small input
TEST(ConvTest, ForwardPass) {
    // Create a 1x1 convolution (dot product) for easy testing
    ConvLayer conv(2, 2, 1, 1, 0);
    
    // Set weights manually
    float weights_data[] = {
        1.0f, 2.0f,  // out_channel 0, in_channels [0,1]
        3.0f, 4.0f   // out_channel 1, in_channels [0,1]
    };
    
    float bias_data[] = {0.1f, 0.2f};
    
    conv.load_weights(weights_data, bias_data);
    
    // Create input: [1, 2, 2, 2]
    // Channel 0: [[1, 2], [3, 4]]
    // Channel 1: [[5, 6], [7, 8]]
    float input_data[] = {
        1.0f, 2.0f, 3.0f, 4.0f,  // batch 0, channel 0
        5.0f, 6.0f, 7.0f, 8.0f   // batch 0, channel 1
    };
    
    Tensor input({1, 2, 2, 2}, input_data, DeviceType::CPU);
    input.to(DeviceType::CUDA);
    
    // Perform forward pass
    Tensor output = conv.forward(input);
    
    // Output should be [1, 2, 2, 2]
    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 2);
    EXPECT_EQ(output.shape()[2], 2);
    EXPECT_EQ(output.shape()[3], 2);
    
    // Move back to CPU to check results
    output.to(DeviceType::CPU);
    
    // Expected results:
    // Channel 0: 1*1 + 5*2 + 0.1 = 11.1, 1*2 + 6*2 + 0.1 = 14.1, ...
    // Channel 1: 1*3 + 5*4 + 0.2 = 23.2, 2*3 + 6*4 + 0.2 = 30.2, ...
    float* out_data = output.data();
    
    // Check a few values (not all for brevity)
    EXPECT_NEAR(out_data[0], 11.1f, 1e-5f);  // batch 0, channel 0, y=0, x=0
    EXPECT_NEAR(out_data[1], 14.1f, 1e-5f);  // batch 0, channel 0, y=0, x=1
    EXPECT_NEAR(out_data[4], 23.2f, 1e-5f);  // batch 0, channel 1, y=0, x=0
    EXPECT_NEAR(out_data[5], 30.2f, 1e-5f);  // batch 0, channel 1, y=0, x=1
}
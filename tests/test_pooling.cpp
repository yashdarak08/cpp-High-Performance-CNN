#include "layers/pooling_layer.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

using namespace cnn_cuda;

// Test max pooling layer
TEST(PoolingTest, MaxPooling) {
    MaxPoolingLayer pool(2, 2);
    
    // Create input: [1, 1, 4, 4]
    float input_data[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    
    Tensor input({1, 1, 4, 4}, input_data, DeviceType::CPU);
    input.to(DeviceType::CUDA);
    
    // Perform forward pass
    Tensor output = pool.forward(input);
    
    // Output should be [1, 1, 2, 2]
    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 1);
    EXPECT_EQ(output.shape()[2], 2);
    EXPECT_EQ(output.shape()[3], 2);
    
    // Move back to CPU to check results
    output.to(DeviceType::CPU);
    
    // Expected max values in each 2x2 block
    float* out_data = output.data();
    EXPECT_FLOAT_EQ(out_data[0], 6.0f);  // max of top-left 2x2
    EXPECT_FLOAT_EQ(out_data[1], 8.0f);  // max of top-right 2x2
    EXPECT_FLOAT_EQ(out_data[2], 14.0f); // max of bottom-left 2x2
    EXPECT_FLOAT_EQ(out_data[3], 16.0f); // max of bottom-right 2x2
}

// Test average pooling layer
TEST(PoolingTest, AvgPooling) {
    AvgPoolingLayer pool(2, 2);
    
    // Create input: [1, 1, 4, 4]
    float input_data[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    
    Tensor input({1, 1, 4, 4}, input_data, DeviceType::CPU);
    input.to(DeviceType::CUDA);
    
    // Perform forward pass
    Tensor output = pool.forward(input);
    
    // Output should be [1, 1, 2, 2]
    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 1);
    EXPECT_EQ(output.shape()[2], 2);
    EXPECT_EQ(output.shape()[3], 2);
    
    // Move back to CPU to check results
    output.to(DeviceType::CPU);
    
    // Expected average values in each 2x2 block
    float* out_data = output.data();
    EXPECT_FLOAT_EQ(out_data[0], 3.5f);  // avg of top-left 2x2
    EXPECT_FLOAT_EQ(out_data[1], 5.5f);  // avg of top-right 2x2
    EXPECT_FLOAT_EQ(out_data[2], 11.5f); // avg of bottom-left 2x2
    EXPECT_FLOAT_EQ(out_data[3], 13.5f); // avg of bottom-right 2x2
}

// Test global average pooling
TEST(PoolingTest, GlobalAvgPooling) {
    // Create input: [1, 2, 4, 4]
    std::vector<float> input_data(32);
    for (int i = 0; i < 32; ++i) {
        input_data[i] = static_cast<float>(i);
    }
    
    Tensor input({1, 2, 4, 4}, input_data.data(), DeviceType::CPU);
    input.to(DeviceType::CUDA);
    
    // Use external helper function to test global average pooling
    extern void launch_global_avg_pool(
        const float* input,
        float* output,
        int batch_size,
        int channels,
        int height,
        int width,
        cudaStream_t stream);
    
    // Create output tensor
    Tensor output({1, 2, 1, 1});
    
    // Perform global average pooling
    launch_global_avg_pool(
        input.data(),
        output.data(),
        1, 2, 4, 4,
        0 // Default stream
    );
    
    // Move back to CPU to check results
    output.to(DeviceType::CPU);
    
    // Expected average values for each channel
    float* out_data = output.data();
    
    // Channel 0: average of 0, 1, 2, ..., 15 = 7.5
    EXPECT_FLOAT_EQ(out_data[0], 7.5f);
    
    // Channel 1: average of 16, 17, 18, ..., 31 = 23.5
    EXPECT_FLOAT_EQ(out_data[1], 23.5f);
}
#include "tensor.h"
#include <gtest/gtest.h>
#include <vector>

using namespace cnn_cuda;

// Test tensor creation and basic operations
TEST(TensorTest, Creation) {
    // Create a tensor on CPU
    Tensor t1({2, 3, 4}, DeviceType::CPU);
    EXPECT_EQ(t1.size(), 24);
    EXPECT_EQ(t1.shape().size(), 3);
    EXPECT_EQ(t1.shape()[0], 2);
    EXPECT_EQ(t1.shape()[1], 3);
    EXPECT_EQ(t1.shape()[2], 4);
    
    // Create a tensor with data
    std::vector<float> data(24, 1.0f);
    Tensor t2({2, 3, 4}, data.data(), DeviceType::CPU);
    EXPECT_EQ(t2.size(), 24);
    
    // Check data access
    float* ptr = t2.data();
    for (int i = 0; i < 24; ++i) {
        EXPECT_FLOAT_EQ(ptr[i], 1.0f);
    }
}

// Test tensor reshape
TEST(TensorTest, Reshape) {
    Tensor t({2, 3, 4}, DeviceType::CPU);
    EXPECT_EQ(t.size(), 24);
    
    // Reshape to different dimensions
    t.reshape({4, 6});
    EXPECT_EQ(t.size(), 24);
    EXPECT_EQ(t.shape().size(), 2);
    EXPECT_EQ(t.shape()[0], 4);
    EXPECT_EQ(t.shape()[1], 6);
    
    // Invalid reshape should throw
    EXPECT_THROW(t.reshape({5, 5}), std::runtime_error);
}

// Test tensor device transfer
TEST(TensorTest, DeviceTransfer) {
    // Create a tensor on CPU with data
    std::vector<float> data(24);
    for (int i = 0; i < 24; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    Tensor cpu_tensor({2, 3, 4}, data.data(), DeviceType::CPU);
    
    // Move to GPU
    cpu_tensor.to(DeviceType::CUDA);
    
    // Move back to CPU to check data
    cpu_tensor.to(DeviceType::CPU);
    
    // Check data integrity
    float* ptr = cpu_tensor.data();
    for (int i = 0; i < 24; ++i) {
        EXPECT_FLOAT_EQ(ptr[i], static_cast<float>(i));
    }
}

// Test tensor operations
TEST(TensorTest, Operations) {
    // Create two tensors with data
    std::vector<float> data1(4, 2.0f);
    std::vector<float> data2(4, 3.0f);
    
    Tensor t1({2, 2}, data1.data(), DeviceType::CPU);
    Tensor t2({2, 2}, data2.data(), DeviceType::CPU);
    
    // Move to GPU
    t1.to(DeviceType::CUDA);
    t2.to(DeviceType::CUDA);
    
    // Add
    Tensor t3({2, 2});
    tensor_add(t1, t2, t3);
    
    // Move back to CPU to check results
    t3.to(DeviceType::CPU);
    
    float* ptr = t3.data();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(ptr[i], 5.0f);
    }
}
#include "cnn_cuda.h"
#include "utils/image_utils.h"
#include "layers/conv_layer.h"
#include "layers/pooling_layer.h"
#include "layers/activation_layer.h"
#include "layers/norm_layer.h"
#include "test_utils.h"
#include <gtest/gtest.h>
#include <memory>

using namespace cnn_cuda;
using namespace cnn_cuda::test;

class CNNPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize the library before each test
        ASSERT_TRUE(cnn_cuda::init());
    }
    
    void TearDown() override {
        // Clean up after each test
        cnn_cuda::shutdown();
    }
    
    TestDataGenerator generator{42};  // Fixed seed for reproducibility
};

// Test a simple CNN forward pass
TEST_F(CNNPipelineTest, SimpleForwardPass) {
    // Create input tensor
    std::vector<int> input_shape = {1, 3, 32, 32};
    Tensor input = generator.generateRandomTensor(input_shape, DeviceType::CUDA, 0.0f, 1.0f);
    
    // Create a simple CNN model
    ConvLayer conv1(3, 16, 3, 1, 1);
    MaxPoolingLayer pool1(2, 2);
    BatchNormLayer bn1(16);
    ConvLayer conv2(16, 32, 3, 1, 1);
    AvgPoolingLayer pool2(2, 2);
    
    // Initialize parameters
    conv1.initialize_parameters();
    conv2.initialize_parameters();
    
    // Forward pass
    Tensor out = conv1.forward(input);
    EXPECT_EQ(out.shape(), std::vector<int>({1, 16, 32, 32}));
    EXPECT_FALSE(hasNaN(out));
    
    out = pool1.forward(out);
    EXPECT_EQ(out.shape(), std::vector<int>({1, 16, 16, 16}));
    EXPECT_FALSE(hasNaN(out));
    
    out = bn1.forward(out);
    EXPECT_EQ(out.shape(), std::vector<int>({1, 16, 16, 16}));
    EXPECT_FALSE(hasNaN(out));
    
    out = relu(out);
    EXPECT_EQ(out.shape(), std::vector<int>({1, 16, 16, 16}));
    EXPECT_FALSE(hasNaN(out));
    
    out = conv2.forward(out);
    EXPECT_EQ(out.shape(), std::vector<int>({1, 32, 16, 16}));
    EXPECT_FALSE(hasNaN(out));
    
    out = pool2.forward(out);
    EXPECT_EQ(out.shape(), std::vector<int>({1, 32, 8, 8}));
    EXPECT_FALSE(hasNaN(out));
    
    // Final output should be valid
    EXPECT_EQ(out.size(), 1 * 32 * 8 * 8);
}

// Test with batch processing
TEST_F(CNNPipelineTest, BatchProcessing) {
    // Create batch input tensor
    int batch_size = 8;
    std::vector<int> input_shape = {batch_size, 3, 64, 64};
    Tensor input = generator.generateRandomTensor(input_shape, DeviceType::CUDA, 0.0f, 1.0f);
    
    // Create a CNN model
    ConvLayer conv1(3, 32, 3, 1, 1);
    MaxPoolingLayer pool1(2, 2);
    BatchNormLayer bn1(32);
    ConvLayer conv2(32, 64, 3, 1, 1);
    AvgPoolingLayer pool2(2, 2);
    ConvLayer conv3(64, 128, 3, 1, 1);
    
    // Initialize parameters
    conv1.initialize_parameters();
    conv2.initialize_parameters();
    conv3.initialize_parameters();
    
    // Forward pass
    Timer timer;
    
    Tensor out = conv1.forward(input);
    out = pool1.forward(out);
    out = bn1.forward(out);
    out = relu(out);
    out = conv2.forward(out);
    out = pool2.forward(out);
    out = relu(out);
    out = conv3.forward(out);
    
    double elapsed = timer.elapsedMilliseconds();
    
    // Check output shape
    std::vector<int> expected_shape = {batch_size, 128, 16, 16};
    EXPECT_EQ(out.shape(), expected_shape);
    EXPECT_FALSE(hasNaN(out));
    
    std::cout << "Batch processing time: " << elapsed << " ms" << std::endl;
}

// Test different convolution implementations
TEST_F(CNNPipelineTest, ConvolutionImplementations) {
    // Create input tensor
    std::vector<int> input_shape = {1, 16, 64, 64};
    Tensor input = generator.generateRandomTensor(input_shape, DeviceType::CUDA);
    
    // Test direct convolution
    {
        ConvLayer conv(16, 32, 3, 1, 1);
        conv.initialize_parameters();
        
        Timer timer;
        Tensor out = conv.forward(input);
        double elapsed = timer.elapsedMilliseconds();
        
        EXPECT_EQ(out.shape(), std::vector<int>({1, 32, 64, 64}));
        EXPECT_FALSE(hasNaN(out));
        
        std::cout << "Direct convolution time: " << elapsed << " ms" << std::endl;
    }
    
    // Test im2col + GEMM convolution
    {
        // Use larger channels to trigger im2col implementation
        ConvLayer conv(16, 64, 5, 1, 2);
        conv.initialize_parameters();
        
        Timer timer;
        Tensor out = conv.forward(input);
        double elapsed = timer.elapsedMilliseconds();
        
        EXPECT_EQ(out.shape(), std::vector<int>({1, 64, 64, 64}));
        EXPECT_FALSE(hasNaN(out));
        
        std::cout << "Im2col+GEMM convolution time: " << elapsed << " ms" << std::endl;
    }
    
    // Test Winograd convolution
    {
        // Use 3x3 kernel with stride 1 to trigger Winograd
        ConvLayer conv(16, 64, 3, 1, 1);
        conv.initialize_parameters();
        
        Timer timer;
        Tensor out = conv.forward(input);
        double elapsed = timer.elapsedMilliseconds();
        
        EXPECT_EQ(out.shape(), std::vector<int>({1, 64, 64, 64}));
        EXPECT_FALSE(hasNaN(out));
        
        std::cout << "Winograd convolution time: " << elapsed << " ms" << std::endl;
    }
}

// Test memory efficiency
TEST_F(CNNPipelineTest, MemoryEfficiency) {
    // Create a larger model to test memory management
    const int batch_size = 16;
    const int input_size = 224;
    
    std::vector<int> input_shape = {batch_size, 3, input_size, input_size};
    Tensor input = generator.generateRandomTensor(input_shape, DeviceType::CUDA);
    
    // Get initial memory usage
    size_t initial_memory = MemoryPool::instance().get_device_memory_usage();
    
    // Process through multiple layers
    {
        ConvLayer conv1(3, 64, 7, 2, 3);
        MaxPoolingLayer pool1(3, 2);
        BatchNormLayer bn1(64);
        
        conv1.initialize_parameters();
        
        Tensor out = conv1.forward(input);
        out = pool1.forward(out);
        out = bn1.forward(out);
        out = relu(out);
        
        // Intermediate size check
        EXPECT_EQ(out.shape(), std::vector<int>({batch_size, 64, 56, 56}));
    }
    
    // Force cleanup
    cnn_cuda::shutdown();
    cnn_cuda::init();
    
    // Check memory usage after cleanup
    size_t final_memory = MemoryPool::instance().get_device_memory_usage();
    
    // Memory should be released
    EXPECT_LT(final_memory, initial_memory * 0.5);
    std::cout << "Memory usage: initial=" << initial_memory / (1024*1024) 
              << "MB, final=" << final_memory / (1024*1024) << "MB" << std::endl;
}

// Test for identity operations
TEST_F(CNNPipelineTest, IdentityOperations) {
    // Create input tensor
    std::vector<int> input_shape = {1, 3, 32, 32};
    Tensor input = generator.generateRandomTensor(input_shape, DeviceType::CUDA);
    Tensor input_cpu = input;
    input_cpu.to(DeviceType::CPU);
    
    // Test identity convolution
    {
        // Create identity kernel where output = input
        Tensor kernel = generator.generateIdentityKernel(3, 3, DeviceType::CPU);
        Tensor bias({3}, DeviceType::CPU);
        
        // Initialize with zeros
        float* bias_data = bias.data();
        for (int i = 0; i < 3; ++i) {
            bias_data[i] = 0.0f;
        }
        
        ConvLayer conv(3, 3, 3, 1, 1);
        conv.load_weights(kernel.data(), bias.data());
        
        Tensor out = conv.forward(input);
        
        // Output should be very close to input (allow for floating point error)
        out.to(DeviceType::CPU);
        expectTensorsNear(input_cpu, out, 1e-4f);
    }
    
    // Test identity batch norm
    {
        BatchNormLayer bn(3);
        
        // Default initialization should be identity-like
        Tensor out = bn.forward(input);
        
        // Output distribution should be normalized but not identical to input
        out.to(DeviceType::CPU);
        
        // Check output is not NaN
        EXPECT_FALSE(hasNaN(out));
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
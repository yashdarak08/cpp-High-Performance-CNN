#include "tensor.h"
#include "test_utils.h"
#include <gtest/gtest.h>

using namespace cnn_cuda;
using namespace cnn_cuda::test;

class TensorTest : public ::testing::Test {
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

// Test tensor creation and basic operations
TEST_F(TensorTest, CreateEmptyTensor) {
    Tensor t;
    EXPECT_EQ(t.size(), 0);
    EXPECT_TRUE(t.shape().empty());
    EXPECT_EQ(t.data(), nullptr);
}

TEST_F(TensorTest, CreateCPUTensor) {
    std::vector<int> shape = {2, 3, 4};
    Tensor t(shape, DeviceType::CPU);
    
    EXPECT_EQ(t.size(), 24);
    EXPECT_EQ(t.shape(), shape);
    EXPECT_NE(t.data(), nullptr);
    EXPECT_EQ(t.device_type(), DeviceType::CPU);
}

TEST_F(TensorTest, CreateGPUTensor) {
    std::vector<int> shape = {2, 3, 4};
    Tensor t(shape, DeviceType::CUDA);
    
    EXPECT_EQ(t.size(), 24);
    EXPECT_EQ(t.shape(), shape);
    EXPECT_NE(t.data(), nullptr);
    EXPECT_EQ(t.device_type(), DeviceType::CUDA);
}

TEST_F(TensorTest, CreateWithData) {
    std::vector<int> shape = {2, 3};
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    
    Tensor t(shape, data.data(), DeviceType::CPU);
    
    EXPECT_EQ(t.size(), 6);
    EXPECT_EQ(t.shape(), shape);
    
    // Check data
    const float* t_data = t.data();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(t_data[i], data[i]);
    }
}

TEST_F(TensorTest, CopyFromCPUToGPU) {
    std::vector<int> shape = {2, 3};
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    
    Tensor cpu_tensor(shape, data.data(), DeviceType::CPU);
    Tensor gpu_tensor(shape, DeviceType::CUDA);
    
    // Copy CPU -> GPU
    gpu_tensor.copy_from(cpu_tensor);
    
    // Copy back to verify
    Tensor copy_back(shape, DeviceType::CPU);
    copy_back.copy_from(gpu_tensor);
    
    // Check data
    const float* copy_data = copy_back.data();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(copy_data[i], data[i]);
    }
}

TEST_F(TensorTest, CopyFromGPUToGPU) {
    std::vector<int> shape = {2, 3};
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    
    Tensor cpu_tensor(shape, data.data(), DeviceType::CPU);
    Tensor gpu_tensor1(shape, DeviceType::CUDA);
    Tensor gpu_tensor2(shape, DeviceType::CUDA);
    
    // Copy CPU -> GPU1
    gpu_tensor1.copy_from(cpu_tensor);
    
    // Copy GPU1 -> GPU2
    gpu_tensor2.copy_from(gpu_tensor1);
    
    // Copy back to verify
    Tensor copy_back(shape, DeviceType::CPU);
    copy_back.copy_from(gpu_tensor2);
    
    // Check data
    const float* copy_data = copy_back.data();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(copy_data[i], data[i]);
    }
}

TEST_F(TensorTest, DeviceTransfer) {
    std::vector<int> shape = {2, 3};
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    
    Tensor tensor(shape, data.data(), DeviceType::CPU);
    
    // Move to GPU
    tensor.to(DeviceType::CUDA);
    EXPECT_EQ(tensor.device_type(), DeviceType::CUDA);
    
    // Move back to CPU
    tensor.to(DeviceType::CPU);
    EXPECT_EQ(tensor.device_type(), DeviceType::CPU);
    
    // Check data is preserved
    const float* tensor_data = tensor.data();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(tensor_data[i], data[i]);
    }
}

TEST_F(TensorTest, Reshape) {
    std::vector<int> shape = {2, 3};
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    
    Tensor tensor(shape, data.data(), DeviceType::CPU);
    
    // Reshape to different dimensions
    std::vector<int> new_shape = {3, 2};
    tensor.reshape(new_shape);
    
    EXPECT_EQ(tensor.shape(), new_shape);
    EXPECT_EQ(tensor.size(), 6);  // Size should remain the same
    
    // Data should be unchanged
    const float* tensor_data = tensor.data();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(tensor_data[i], data[i]);
    }
    
    // Try invalid reshape
    EXPECT_THROW(tensor.reshape({4, 2}), std::runtime_error);
}

TEST_F(TensorTest, TensorAddCPU) {
    std::vector<int> shape = {2, 3};
    std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b_data = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    std::vector<float> expected = {7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f};
    
    Tensor a(shape, a_data.data(), DeviceType::CPU);
    Tensor b(shape, b_data.data(), DeviceType::CPU);
    Tensor result(shape, DeviceType::CPU);
    
    tensor_add(a, b, result);
    
    // Check result
    const float* result_data = result.data();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(result_data[i], expected[i]);
    }
}

TEST_F(TensorTest, TensorAddGPU) {
    std::vector<int> shape = {2, 3};
    std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b_data = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    std::vector<float> expected = {7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f};
    
    Tensor a(shape, a_data.data(), DeviceType::CPU);
    Tensor b(shape, b_data.data(), DeviceType::CPU);
    Tensor result(shape, DeviceType::CPU);
    
    // Move to GPU
    a.to(DeviceType::CUDA);
    b.to(DeviceType::CUDA);
    result.to(DeviceType::CUDA);
    
    tensor_add(a, b, result);
    
    // Move back to CPU
    result.to(DeviceType::CPU);
    
    // Check result
    const float* result_data = result.data();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(result_data[i], expected[i]);
    }
}

TEST_F(TensorTest, TensorMulCPU) {
    std::vector<int> shape = {2, 3};
    std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b_data = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    std::vector<float> expected = {6.0f, 10.0f, 12.0f, 12.0f, 10.0f, 6.0f};
    
    Tensor a(shape, a_data.data(), DeviceType::CPU);
    Tensor b(shape, b_data.data(), DeviceType::CPU);
    Tensor result(shape, DeviceType::CPU);
    
    tensor_mul(a, b, result);
    
    // Check result
    const float* result_data = result.data();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(result_data[i], expected[i]);
    }
}

TEST_F(TensorTest, TensorMulGPU) {
    std::vector<int> shape = {2, 3};
    std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b_data = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    std::vector<float> expected = {6.0f, 10.0f, 12.0f, 12.0f, 10.0f, 6.0f};
    
    Tensor a(shape, a_data.data(), DeviceType::CPU);
    Tensor b(shape, b_data.data(), DeviceType::CPU);
    Tensor result(shape, DeviceType::CPU);
    
    // Move to GPU
    a.to(DeviceType::CUDA);
    b.to(DeviceType::CUDA);
    result.to(DeviceType::CUDA);
    
    tensor_mul(a, b, result);
    
    // Move back to CPU
    result.to(DeviceType::CPU);
    
    // Check result
    const float* result_data = result.data();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(result_data[i], expected[i]);
    }
}

TEST_F(TensorTest, TensorReLUCPU) {
    std::vector<int> shape = {2, 3};
    std::vector<float> input_data = {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f};
    std::vector<float> expected = {0.0f, 2.0f, 0.0f, 4.0f, 0.0f, 6.0f};
    
    Tensor input(shape, input_data.data(), DeviceType::CPU);
    Tensor result(shape, DeviceType::CPU);
    
    tensor_relu(input, result);
    
    // Check result
    const float* result_data = result.data();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(result_data[i], expected[i]);
    }
}

TEST_F(TensorTest, TensorReLUGPU) {
    std::vector<int> shape = {2, 3};
    std::vector<float> input_data = {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f};
    std::vector<float> expected = {0.0f, 2.0f, 0.0f, 4.0f, 0.0f, 6.0f};
    
    Tensor input(shape, input_data.data(), DeviceType::CPU);
    Tensor result(shape, DeviceType::CPU);
    
    // Move to GPU
    input.to(DeviceType::CUDA);
    result.to(DeviceType::CUDA);
    
    tensor_relu(input, result);
    
    // Move back to CPU
    result.to(DeviceType::CPU);
    
    // Check result
    const float* result_data = result.data();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(result_data[i], expected[i]);
    }
}

TEST_F(TensorTest, RandomTensorGeneration) {
    std::vector<int> shape = {10, 10};
    Tensor t = generator.generateRandomTensor(shape, DeviceType::CPU, -1.0f, 1.0f);
    
    EXPECT_EQ(t.size(), 100);
    EXPECT_EQ(t.shape(), shape);
    
    // Check that values are within range
    const float* data = t.data();
    for (int i = 0; i < 100; ++i) {
        EXPECT_GE(data[i], -1.0f);
        EXPECT_LE(data[i], 1.0f);
    }
}

TEST_F(TensorTest, LargeTensor) {
    // Test with a larger tensor to check for memory issues
    std::vector<int> shape = {32, 64, 128, 128};
    Tensor t(shape, DeviceType::CUDA);
    
    EXPECT_EQ(t.size(), 32 * 64 * 128 * 128);
    EXPECT_EQ(t.shape(), shape);
}

// Performance test for tensor operations
TEST_F(TensorTest, PerformanceTest) {
    // Skip in regular test runs
    if (::testing::FLAGS_gtest_filter != "*PerformanceTest") {
        GTEST_SKIP();
    }
    
    std::vector<int> shape = {128, 256, 256};
    Tensor a = generator.generateRandomTensor(shape, DeviceType::CUDA);
    Tensor b = generator.generateRandomTensor(shape, DeviceType::CUDA);
    Tensor result(shape, DeviceType::CUDA);
    
    const int num_runs = 100;
    
    // Warm-up
    for (int i = 0; i < 10; ++i) {
        tensor_add(a, b, result);
    }
    
    // Benchmark tensor_add
    Timer timer;
    for (int i = 0; i < num_runs; ++i) {
        tensor_add(a, b, result);
    }
    double add_time = timer.elapsedMilliseconds() / num_runs;
    
    // Benchmark tensor_mul
    timer.reset();
    for (int i = 0; i < num_runs; ++i) {
        tensor_mul(a, b, result);
    }
    double mul_time = timer.elapsedMilliseconds() / num_runs;
    
    // Benchmark tensor_relu
    timer.reset();
    for (int i = 0; i < num_runs; ++i) {
        tensor_relu(a, result);
    }
    double relu_time = timer.elapsedMilliseconds() / num_runs;
    
    std::cout << "Performance results for tensor operations (ms/op):" << std::endl;
    std::cout << "  tensor_add: " << add_time << std::endl;
    std::cout << "  tensor_mul: " << mul_time << std::endl;
    std::cout << "  tensor_relu: " << relu_time << std::endl;
}

// Main function for the test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
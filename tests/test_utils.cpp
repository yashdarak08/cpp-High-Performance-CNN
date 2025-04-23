// tests/test_utils.cpp
#include "test_utils.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace cnn_cuda {
namespace test {

TestDataGenerator::TestDataGenerator(unsigned int seed) : rng_(seed) {}

Tensor TestDataGenerator::generateRandomTensor(const std::vector<int>& shape, 
                                             DeviceType device,
                                             float min_val,
                                             float max_val) {
    int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    std::vector<float> data(size);
    
    std::uniform_real_distribution<float> dist(min_val, max_val);
    std::generate(data.begin(), data.end(), [&]() { return dist(rng_); });
    
    return Tensor(shape, data.data(), device);
}

Tensor TestDataGenerator::generateZeroTensor(const std::vector<int>& shape, 
                                           DeviceType device) {
    int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    std::vector<float> data(size, 0.0f);
    
    return Tensor(shape, data.data(), device);
}

Tensor TestDataGenerator::generateConstantTensor(const std::vector<int>& shape, 
                                               float value,
                                               DeviceType device) {
    int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    std::vector<float> data(size, value);
    
    return Tensor(shape, data.data(), device);
}

Tensor TestDataGenerator::generateIdentityKernel(int channels, int kernel_size,
                                               DeviceType device) {
    // Create a kernel that preserves the input (identity convolution)
    std::vector<int> shape = {channels, channels, kernel_size, kernel_size};
    int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    std::vector<float> data(size, 0.0f);
    
    // Set center value to 1.0 for each input channel
    int center = kernel_size / 2;
    for (int i = 0; i < channels; ++i) {
        int idx = ((i * channels + i) * kernel_size + center) * kernel_size + center;
        data[idx] = 1.0f;
    }
    
    return Tensor(shape, data.data(), device);
}

void expectTensorsNear(const Tensor& expected, const Tensor& actual, float tolerance) {
    // Check shape
    ASSERT_EQ(expected.shape(), actual.shape()) << "Tensor shapes don't match";
    
    // Move both tensors to CPU for comparison
    Tensor expected_cpu = expected;
    Tensor actual_cpu = actual;
    
    if (expected.device_type() == DeviceType::CUDA) {
        expected_cpu.to(DeviceType::CPU);
    }
    
    if (actual.device_type() == DeviceType::CUDA) {
        actual_cpu.to(DeviceType::CPU);
    }
    
    const float* expected_data = expected_cpu.data();
    const float* actual_data = actual_cpu.data();
    
    // Compare element-wise
    for (int i = 0; i < expected_cpu.size(); ++i) {
        ASSERT_NEAR(expected_data[i], actual_data[i], tolerance)
            << "Tensors differ at index " << i;
    }
}

bool hasNaN(const Tensor& tensor) {
    // Move tensor to CPU for checking
    Tensor cpu_tensor = tensor;
    if (tensor.device_type() == DeviceType::CUDA) {
        cpu_tensor.to(DeviceType::CPU);
    }
    
    const float* data = cpu_tensor.data();
    
    for (int i = 0; i < cpu_tensor.size(); ++i) {
        if (std::isnan(data[i])) {
            return true;
        }
    }
    
    return false;
}

std::vector<float> loadTestData(const std::string& filename) {
    std::string full_path = std::string(TEST_DATA_DIR) + "/" + filename;
    std::ifstream file(full_path, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open test data file: " + full_path);
    }
    
    // Read file size
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Read data
    std::vector<float> data(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    
    return data;
}

void saveTensorToFile(const Tensor& tensor, const std::string& filename) {
    // Move tensor to CPU
    Tensor cpu_tensor = tensor;
    if (tensor.device_type() == DeviceType::CUDA) {
        cpu_tensor.to(DeviceType::CPU);
    }
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    
    // Write tensor data
    const float* data = cpu_tensor.data();
    file.write(reinterpret_cast<const char*>(data), cpu_tensor.size() * sizeof(float));
}

} // namespace test
} // namespace cnn_cuda
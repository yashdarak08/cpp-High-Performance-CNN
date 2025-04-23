// tests/test_utils.h
#pragma once

#include "tensor.h"
#include "gtest/gtest.h"
#include <vector>
#include <string>
#include <random>
#include <chrono>

namespace cnn_cuda {
namespace test {

// Tolerance for floating-point comparisons
constexpr float DEFAULT_TOLERANCE = 1e-5f;

// Class to generate test data
class TestDataGenerator {
public:
    TestDataGenerator(unsigned int seed = 0);
    
    // Generate random tensors
    Tensor generateRandomTensor(const std::vector<int>& shape, 
                               DeviceType device = DeviceType::CPU,
                               float min_val = -1.0f,
                               float max_val = 1.0f);
    
    // Generate zero/constant tensors
    Tensor generateZeroTensor(const std::vector<int>& shape, 
                             DeviceType device = DeviceType::CPU);
    
    Tensor generateConstantTensor(const std::vector<int>& shape, 
                                 float value,
                                 DeviceType device = DeviceType::CPU);
    
    // Generate identity convolution kernels
    Tensor generateIdentityKernel(int channels, int kernel_size,
                                 DeviceType device = DeviceType::CPU);
    
private:
    std::mt19937 rng_;
};

// Function to compare two tensors element-wise
void expectTensorsNear(const Tensor& expected, const Tensor& actual, 
                      float tolerance = DEFAULT_TOLERANCE);

// Function to check if the tensor has NaN values
bool hasNaN(const Tensor& tensor);

// Utility timer for performance measurements
class Timer {
public:
    Timer() : start_time_(std::chrono::high_resolution_clock::now()) {}
    
    void reset() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsedMilliseconds() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_time_).count();
    }
    
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

// Load test data from file
std::vector<float> loadTestData(const std::string& filename);

// Save tensor to file (for debugging)
void saveTensorToFile(const Tensor& tensor, const std::string& filename);

} // namespace test
} // namespace cnn_cuda
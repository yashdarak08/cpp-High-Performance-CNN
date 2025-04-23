#pragma once

#include <vector>
#include <memory>
#include <cassert>
#include <cuda_runtime.h>

namespace cnn_cuda {

enum class DeviceType {
    CPU,
    CUDA
};

class Tensor {
public:
    Tensor() = default;
    
    // Create tensor with specified dimensions
    Tensor(const std::vector<int>& shape, DeviceType device = DeviceType::CUDA);
    
    // Create tensor with data
    Tensor(const std::vector<int>& shape, const float* data, DeviceType device = DeviceType::CUDA);
    
    // Move data between devices
    void to(DeviceType device);
    
    // Copy data from another tensor
    void copy_from(const Tensor& other);
    
    // Access methods
    float* data();
    const float* data() const;
    
    // Shape access methods
    const std::vector<int>& shape() const;
    int dim(int i) const;
    int size() const; // total number of elements
    
    // Reshape tensor (doesn't modify data)
    void reshape(const std::vector<int>& new_shape);
    
    // Destructor to properly free memory
    ~Tensor();

private:
    std::vector<int> shape_;
    float* data_ptr_ = nullptr;
    size_t size_ = 0;
    DeviceType device_ = DeviceType::CPU;
    
    void allocate();
    void free();
};

// Helper functions for tensor operations
void tensor_add(const Tensor& a, const Tensor& b, Tensor& out);
void tensor_mul(const Tensor& a, const Tensor& b, Tensor& out);
void tensor_relu(const Tensor& input, Tensor& output);

} // namespace cnn_cuda
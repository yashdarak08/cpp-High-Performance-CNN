#include "tensor.h"
#include "utils/memory_pool.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <numeric>
#include <functional>

namespace cnn_cuda {

Tensor::Tensor(const std::vector<int>& shape, DeviceType device) 
    : shape_(shape), device_(device) {
    
    // Calculate total size
    size_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    
    // Allocate memory
    allocate();
}

Tensor::Tensor(const std::vector<int>& shape, const float* data, DeviceType device) 
    : shape_(shape), device_(device) {
    
    // Calculate total size
    size_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    
    // Allocate memory
    allocate();
    
    // Copy data
    if (data != nullptr) {
        if (device_ == DeviceType::CPU) {
            std::memcpy(data_ptr_, data, size_ * sizeof(float));
        } else {
            cudaMemcpy(data_ptr_, data, size_ * sizeof(float), cudaMemcpyHostToDevice);
        }
    }
}

void Tensor::to(DeviceType target_device) {
    if (device_ == target_device) return;
    
    // Allocate memory on target device
    float* new_data = nullptr;
    
    if (target_device == DeviceType::CPU) {
        new_data = static_cast<float*>(MemoryPool::instance().allocate(size_ * sizeof(float), false));
        cudaMemcpy(new_data, data_ptr_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        new_data = static_cast<float*>(MemoryPool::instance().allocate(size_ * sizeof(float), true));
        cudaMemcpy(new_data, data_ptr_, size_ * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    // Free old memory
    MemoryPool::instance().free(data_ptr_);
    
    // Update pointer and device
    data_ptr_ = new_data;
    device_ = target_device;
}

void Tensor::copy_from(const Tensor& other) {
    if (size_ != other.size()) {
        throw std::runtime_error("Cannot copy from tensor of different size");
    }
    
    if (device_ == other.device_) {
        if (device_ == DeviceType::CPU) {
            std::memcpy(data_ptr_, other.data_ptr_, size_ * sizeof(float));
        } else {
            cudaMemcpy(data_ptr_, other.data_ptr_, size_ * sizeof(float), cudaMemcpyDeviceToDevice);
        }
    } else if (device_ == DeviceType::CPU && other.device_ == DeviceType::CUDA) {
        cudaMemcpy(data_ptr_, other.data_ptr_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(data_ptr_, other.data_ptr_, size_ * sizeof(float), cudaMemcpyHostToDevice);
    }
}

float* Tensor::data() {
    return data_ptr_;
}

const float* Tensor::data() const {
    return data_ptr_;
}

const std::vector<int>& Tensor::shape() const {
    return shape_;
}

int Tensor::dim(int i) const {
    if (i < 0 || i >= shape_.size()) {
        throw std::out_of_range("Dimension index out of range");
    }
    return shape_[i];
}

int Tensor::size() const {
    return size_;
}

void Tensor::reshape(const std::vector<int>& new_shape) {
    int new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
    
    if (new_size != size_) {
        throw std::runtime_error("Reshape must preserve total number of elements");
    }
    
    shape_ = new_shape;
}

void Tensor::allocate() {
    if (size_ == 0) return;
    
    if (device_ == DeviceType::CPU) {
        data_ptr_ = static_cast<float*>(MemoryPool::instance().allocate(size_ * sizeof(float), false));
    } else {
        data_ptr_ = static_cast<float*>(MemoryPool::instance().allocate(size_ * sizeof(float), true));
    }
}

void Tensor::free() {
    if (data_ptr_ != nullptr) {
        MemoryPool::instance().free(data_ptr_);
        data_ptr_ = nullptr;
    }
}

Tensor::~Tensor() {
    free();
}

} // namespace cnn_cuda
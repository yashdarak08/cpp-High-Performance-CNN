#include "utils/memory_pool.h"
#include <stdexcept>
#include <algorithm>

namespace cnn_cuda {

MemoryPool& MemoryPool::instance() {
    static MemoryPool instance;
    return instance;
}

MemoryPool::MemoryPool() : device_memory_usage_(0), host_memory_usage_(0) {}

MemoryPool::~MemoryPool() {
    clear();
}

void* MemoryPool::allocate(size_t size, bool device_memory) {
    if (size == 0) return nullptr;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (device_memory) {
        return allocate_device(size);
    } else {
        return allocate_host(size);
    }
}

void MemoryPool::free(void* ptr) {
    if (ptr == nullptr) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check if this is a device or host pointer
    auto dev_it = device_blocks_.find(ptr);
    if (dev_it != device_blocks_.end()) {
        free_device(ptr);
        return;
    }
    
    auto host_it = host_blocks_.find(ptr);
    if (host_it != host_blocks_.end()) {
        free_host(ptr);
        return;
    }
    
    throw std::runtime_error("Attempted to free unknown pointer");
}

void MemoryPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Free all device memory
    for (const auto& pair : device_blocks_) {
        cudaFree(pair.first);
    }
    device_blocks_.clear();
    free_device_blocks_.clear();
    device_memory_usage_ = 0;
    
    // Free all host memory
    for (const auto& pair : host_blocks_) {
        cudaFreeHost(pair.first);
    }
    host_blocks_.clear();
    free_host_blocks_.clear();
    host_memory_usage_ = 0;
}

size_t MemoryPool::get_device_memory_usage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return device_memory_usage_;
}

size_t MemoryPool::get_host_memory_usage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return host_memory_usage_;
}

void* MemoryPool::allocate_device(size_t size) {
    // Check if we have a free block of appropriate size
    auto it = free_device_blocks_.find(size);
    if (it != free_device_blocks_.end() && !it->second.empty()) {
        // Reuse existing block
        void* ptr = it->second.back();
        it->second.pop_back();
        
        // Mark as in use
        device_blocks_[ptr].in_use = true;
        
        return ptr;
    }
    
    // Allocate new block
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    
    if (err != cudaSuccess || ptr == nullptr) {
        throw std::runtime_error("Failed to allocate CUDA memory");
    }
    
    // Register the new block
    device_blocks_[ptr] = {ptr, size, true};
    device_memory_usage_ += size;
    
    return ptr;
}

void* MemoryPool::allocate_host(size_t size) {
    // Check if we have a free block of appropriate size
    auto it = free_host_blocks_.find(size);
    if (it != free_host_blocks_.end() && !it->second.empty()) {
        // Reuse existing block
        void* ptr = it->second.back();
        it->second.pop_back();
        
        // Mark as in use
        host_blocks_[ptr].in_use = true;
        
        return ptr;
    }
    
    // Allocate new block
    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    
    if (err != cudaSuccess || ptr == nullptr) {
        throw std::runtime_error("Failed to allocate pinned host memory");
    }
    
    // Register the new block
    host_blocks_[ptr] = {ptr, size, true};
    host_memory_usage_ += size;
    
    return ptr;
}

void MemoryPool::free_device(void* ptr) {
    auto it = device_blocks_.find(ptr);
    if (it == device_blocks_.end()) {
        throw std::runtime_error("Attempted to free unknown device pointer");
    }
    
    // If already in the free list, do nothing
    if (!it->second.in_use) {
        return;
    }
    
    // Mark as not in use
    it->second.in_use = false;
    
    // Add to free list
    free_device_blocks_[it->second.size].push_back(ptr);
}

void MemoryPool::free_host(void* ptr) {
    auto it = host_blocks_.find(ptr);
    if (it == host_blocks_.end()) {
        throw std::runtime_error("Attempted to free unknown host pointer");
    }
    
    // If already in the free list, do nothing
    if (!it->second.in_use) {
        return;
    }
    
    // Mark as not in use
    it->second.in_use = false;
    
    // Add to free list
    free_host_blocks_[it->second.size].push_back(ptr);
}

} // namespace cnn_cuda
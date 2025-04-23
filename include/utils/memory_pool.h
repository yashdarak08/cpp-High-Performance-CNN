#pragma once

#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>

namespace cnn_cuda {

class MemoryPool {
public:
    // Get singleton instance
    static MemoryPool& instance();
    
    // Allocate memory of specified size
    void* allocate(size_t size, bool device_memory = true);
    
    // Free allocated memory
    void free(void* ptr);
    
    // Free all memory at once
    void clear();
    
    // Get current memory usage stats
    size_t get_device_memory_usage() const;
    size_t get_host_memory_usage() const;
    
private:
    MemoryPool();  // Private constructor for singleton
    ~MemoryPool();
    
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    
    // Memory block structure
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };
    
    // Maps for managing memory blocks
    std::unordered_map<void*, Block> device_blocks_;
    std::unordered_map<void*, Block> host_blocks_;
    
    // Free blocks organized by size for quick allocation
    std::unordered_map<size_t, std::vector<void*>> free_device_blocks_;
    std::unordered_map<size_t, std::vector<void*>> free_host_blocks_;
    
    // Memory usage tracking
    size_t device_memory_usage_;
    size_t host_memory_usage_;
    
    // Thread safety
    mutable std::mutex mutex_;
    
    // Helper methods
    void* allocate_device(size_t size);
    void* allocate_host(size_t size);
    void free_device(void* ptr);
    void free_host(void* ptr);
};

} // namespace cnn_cuda
#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <mutex>

namespace cnn_cuda {

// Error severity levels
enum class ErrorSeverity {
    WARNING,   // Non-critical issues that don't prevent execution
    ERROR,     // Critical issues that prevent execution but may be recoverable
    FATAL      // Unrecoverable errors that require application termination
};

// Error codes
enum class ErrorCode {
    // General errors
    UNKNOWN_ERROR = 0,
    INVALID_ARGUMENT = 1,
    OUT_OF_MEMORY = 2,
    INVALID_DEVICE = 3,
    
    // CUDA-specific errors
    CUDA_ERROR = 100,
    CUDA_MEMORY_ALLOCATION_FAILED = 101,
    CUDA_KERNEL_LAUNCH_FAILED = 102,
    CUDA_DEVICE_NOT_AVAILABLE = 103,
    
    // Tensor errors
    TENSOR_SHAPE_MISMATCH = 200,
    TENSOR_INVALID_OPERATION = 201,
    TENSOR_OUT_OF_BOUNDS = 202,
    
    // Layer errors
    LAYER_INITIALIZATION_FAILED = 300,
    LAYER_INVALID_PARAMETERS = 301,
    LAYER_FORWARD_FAILED = 302,
    
    // I/O errors
    IO_FILE_NOT_FOUND = 400,
    IO_INVALID_FORMAT = 401,
    IO_READ_FAILED = 402,
    IO_WRITE_FAILED = 403
};

// Exception class for CNN CUDA errors
class CNNException : public std::runtime_error {
public:
    CNNException(ErrorCode code, const std::string& message, ErrorSeverity severity = ErrorSeverity::ERROR)
        : std::runtime_error(message), code_(code), severity_(severity) {}
    
    ErrorCode getErrorCode() const { return code_; }
    ErrorSeverity getSeverity() const { return severity_; }
    
private:
    ErrorCode code_;
    ErrorSeverity severity_;
};

// Error handling utility class
class ErrorHandler {
public:
    // Get singleton instance
    static ErrorHandler& instance() {
        static ErrorHandler instance;
        return instance;
    }
    
    // Log an error without throwing
    void logError(ErrorCode code, const std::string& message, 
                 ErrorSeverity severity = ErrorSeverity::ERROR,
                 const char* file = nullptr, int line = -1) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::stringstream ss;
        ss << "[" << getSeverityString(severity) << "] ";
        ss << "Error " << static_cast<int>(code) << ": " << message;
        
        if (file && line >= 0) {
            ss << " (" << file << ":" << line << ")";
        }
        
        error_log_.push_back(ss.str());
        
        // Print to stderr for immediate visibility
        std::cerr << ss.str() << std::endl;
    }
    
    // Check CUDA error and throw if needed
    void checkCudaError(cudaError_t error, const char* file = nullptr, int line = -1) {
        if (error != cudaSuccess) {
            std::stringstream ss;
            ss << "CUDA error: " << cudaGetErrorString(error);
            
            // Map CUDA errors to our error codes
            ErrorCode code = ErrorCode::CUDA_ERROR;
            if (error == cudaErrorMemoryAllocation) {
                code = ErrorCode::CUDA_MEMORY_ALLOCATION_FAILED;
            } else if (error == cudaErrorNoDevice || error == cudaErrorInvalidDevice) {
                code = ErrorCode::CUDA_DEVICE_NOT_AVAILABLE;
            }
            
            logError(code, ss.str(), ErrorSeverity::ERROR, file, line);
            throw CNNException(code, ss.str());
        }
    }
    
    // Get all logged errors
    const std::vector<std::string>& getErrorLog() const {
        return error_log_;
    }
    
    // Clear error log
    void clearErrorLog() {
        std::lock_guard<std::mutex> lock(mutex_);
        error_log_.clear();
    }
    
private:
    ErrorHandler() = default;
    ~ErrorHandler() = default;
    
    ErrorHandler(const ErrorHandler&) = delete;
    ErrorHandler& operator=(const ErrorHandler&) = delete;
    
    std::string getSeverityString(ErrorSeverity severity) {
        switch (severity) {
            case ErrorSeverity::WARNING: return "WARNING";
            case ErrorSeverity::ERROR: return "ERROR";
            case ErrorSeverity::FATAL: return "FATAL";
            default: return "UNKNOWN";
        }
    }
    
    std::vector<std::string> error_log_;
    std::mutex mutex_;
};

// Utility macros for convenient error handling
#define CNN_CUDA_CHECK(call) \
    cnn_cuda::ErrorHandler::instance().checkCudaError(call, __FILE__, __LINE__)

#define CNN_CUDA_LOG_ERROR(code, message, severity) \
    cnn_cuda::ErrorHandler::instance().logError(code, message, severity, __FILE__, __LINE__)

#define CNN_CUDA_THROW_ERROR(code, message, severity) \
    do { \
        cnn_cuda::ErrorHandler::instance().logError(code, message, severity, __FILE__, __LINE__); \
        throw cnn_cuda::CNNException(code, message, severity); \
    } while (0)

// Memory recovery functions
// Use to clean up resources when errors occur

// Safely free CUDA memory
template<typename T>
void safeCudaFree(T* &ptr) {
    if (ptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

// Safe tensor cleanup
void cleanupTensors(std::vector<Tensor*>& tensors) {
    for (auto tensor : tensors) {
        delete tensor;
    }
    tensors.clear();
}

} // namespace cnn_cuda
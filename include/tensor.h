#pragma once

#include <vector>
#include <memory>
#include <cassert>
#include <cuda_runtime.h>
#include <string>
#include <stdexcept>

namespace cnn_cuda {

/**
 * @enum DeviceType
 * @brief Enumeration of supported compute devices
 */
enum class DeviceType {
    CPU,    /**< Host CPU memory */
    CUDA    /**< NVIDIA GPU CUDA device memory */
};

/**
 * @enum DataType
 * @brief Enumeration of supported data types
 */
enum class DataType {
    FLOAT32,    /**< 32-bit floating point */
    FLOAT16,    /**< 16-bit floating point (half precision) */
    INT8,       /**< 8-bit integer (quantized) */
    UINT8       /**< Unsigned 8-bit integer */
};

/**
 * @class Tensor
 * @brief Multi-dimensional array with automatic memory management and device transfer
 */
class Tensor {
public:
    /**
     * @brief Default constructor
     */
    Tensor() = default;
    
    /**
     * @brief Create tensor with specified dimensions and data type
     * @param shape Vector of dimensions
     * @param device Device where the tensor should be allocated
     * @param dtype Data type for the tensor elements
     */
    Tensor(const std::vector<int>& shape, 
           DeviceType device = DeviceType::CUDA,
           DataType dtype = DataType::FLOAT32);
    
    /**
     * @brief Create tensor with data
     * @param shape Vector of dimensions
     * @param data Pointer to initial data (must be float32 for now)
     * @param device Device where the tensor should be allocated
     * @param dtype Data type for the tensor elements
     */
    Tensor(const std::vector<int>& shape, 
           const float* data, 
           DeviceType device = DeviceType::CUDA,
           DataType dtype = DataType::FLOAT32);
    
    /**
     * @brief Create tensor with int8 data
     * @param shape Vector of dimensions
     * @param data Pointer to initial int8 data
     * @param device Device where the tensor should be allocated
     */
    Tensor(const std::vector<int>& shape, 
           const int8_t* data, 
           DeviceType device = DeviceType::CUDA);
    
    /**
     * @brief Create tensor with uint8 data
     * @param shape Vector of dimensions
     * @param data Pointer to initial uint8 data
     * @param device Device where the tensor should be allocated
     */
    Tensor(const std::vector<int>& shape, 
           const uint8_t* data, 
           DeviceType device = DeviceType::CUDA);
    
    /**
     * @brief Move data between devices
     * @param device Target device (CPU or CUDA)
     */
    void to(DeviceType device);
    
    /**
     * @brief Convert tensor to a different data type
     * @param dtype Target data type
     * @return A new tensor with the converted data type
     */
    Tensor to_dtype(DataType dtype) const;
    
    /**
     * @brief Copy data from another tensor
     * @param other Source tensor
     */
    void copy_from(const Tensor& other);
    
    /**
     * @brief Get raw pointer to tensor data
     * @return Pointer to the underlying data
     */
    void* data();
    
    /**
     * @brief Get const raw pointer to tensor data
     * @return Const pointer to the underlying data
     */
    const void* data() const;
    
    /**
     * @brief Get typed pointer to tensor data (float32)
     * @return Typed pointer to the data (or nullptr if type mismatch)
     */
    float* data_f32();
    const float* data_f32() const;
    
    /**
     * @brief Get typed pointer to tensor data (int8)
     * @return Typed pointer to the data (or nullptr if type mismatch)
     */
    int8_t* data_i8();
    const int8_t* data_i8() const;
    
    /**
     * @brief Get typed pointer to tensor data (uint8)
     * @return Typed pointer to the data (or nullptr if type mismatch)
     */
    uint8_t* data_u8();
    const uint8_t* data_u8() const;
    
    /**
     * @brief Get the tensor's shape
     * @return Vector of dimensions
     */
    const std::vector<int>& shape() const;
    
    /**
     * @brief Get a specific dimension size
     * @param i Dimension index
     * @return Size of the specified dimension
     */
    int dim(int i) const;
    
    /**
     * @brief Get the total number of elements
     * @return Total number of elements in the tensor
     */
    int size() const;
    
    /**
     * @brief Get the size in bytes of a single element
     * @return Element size in bytes
     */
    int element_size() const;
    
    /**
     * @brief Get the total size in bytes
     * @return Total memory size of the tensor in bytes
     */
    size_t bytes() const;
    
    /**
     * @brief Reshape tensor (doesn't modify data)
     * @param new_shape New dimensions
     */
    void reshape(const std::vector<int>& new_shape);
    
    /**
     * @brief Get the current device type
     * @return The device where the tensor data resides
     */
    DeviceType device_type() const;
    
    /**
     * @brief Get the data type
     * @return Data type of the tensor elements
     */
    DataType data_type() const;
    
    /**
     * @brief Get a string representation of the tensor
     * @param max_elements Maximum number of elements to include in the string
     * @return String representation
     */
    std::string to_string(int max_elements = 100) const;
    
    /**
     * @brief Print tensor to standard output
     * @param max_elements Maximum number of elements to print
     */
    void print(int max_elements = 100) const;
    
    /**
     * @brief Destructor to properly free memory
     */
    ~Tensor();

private:
    std::vector<int> shape_;      /**< Shape of the tensor */
    void* data_ptr_ = nullptr;    /**< Pointer to the data */
    size_t size_ = 0;             /**< Total number of elements */
    DeviceType device_ = DeviceType::CPU;  /**< Current device */
    DataType dtype_ = DataType::FLOAT32;   /**< Data type */
    
    /**
     * @brief Allocate memory for the tensor
     */
    void allocate();
    
    /**
     * @brief Free the tensor's memory
     */
    void free();
    
    /**
     * @brief Get the size in bytes for a data type
     * @param dtype Data type
     * @return Size in bytes
     */
    static int get_element_size(DataType dtype);
};

/**
 * @brief Add two tensors element-wise
 * @param a First tensor
 * @param b Second tensor
 * @param out Output tensor
 */
void tensor_add(const Tensor& a, const Tensor& b, Tensor& out);

/**
 * @brief Multiply two tensors element-wise
 * @param a First tensor
 * @param b Second tensor
 * @param out Output tensor
 */
void tensor_mul(const Tensor& a, const Tensor& b, Tensor& out);

/**
 * @brief Apply ReLU activation function to a tensor
 * @param input Input tensor
 * @param output Output tensor
 */
void tensor_relu(const Tensor& input, Tensor& output);

/**
 * @brief Quantize a float32 tensor to int8
 * @param input Input float32 tensor
 * @param scale Scale factor for quantization
 * @param zero_point Zero point for quantization
 * @return Quantized int8 tensor
 */
Tensor quantize_to_int8(const Tensor& input, float scale, int8_t zero_point);

/**
 * @brief Dequantize an int8 tensor to float32
 * @param input Input int8 tensor
 * @param scale Scale factor for dequantization
 * @param zero_point Zero point for dequantization
 * @return Dequantized float32 tensor
 */
Tensor dequantize_to_float32(const Tensor& input, float scale, int8_t zero_point);

} // namespace cnn_cuda
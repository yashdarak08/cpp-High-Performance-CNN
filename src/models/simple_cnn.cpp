// This implementation goes in src/model/simple_cnn.cpp

#include "examples/simple_cnn.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace cnn_cuda {

void SimpleCNN::load_weights(const std::string& weights_file) {
    std::ifstream file(weights_file, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open weights file: " + weights_file);
    }
    
    std::cout << "Loading weights from " << weights_file << std::endl;
    
    // Read number of layers
    int num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    
    // Read header info
    int model_version;
    file.read(reinterpret_cast<char*>(&model_version), sizeof(model_version));
    
    // Check if the file has the correct number of layers
    if (num_layers != 7) {  // 3 conv layers + 1 FC layer + 3 batch norm layers
        throw std::runtime_error("Weight file has incorrect number of layers");
    }
    
    // Load conv1 weights
    {
        int in_channels, out_channels, kernel_size;
        file.read(reinterpret_cast<char*>(&in_channels), sizeof(in_channels));
        file.read(reinterpret_cast<char*>(&out_channels), sizeof(out_channels));
        file.read(reinterpret_cast<char*>(&kernel_size), sizeof(kernel_size));
        
        if (in_channels != 3 || out_channels != 32 || kernel_size != 3) {
            throw std::runtime_error("Conv1 parameters mismatch");
        }
        
        int weights_size = in_channels * out_channels * kernel_size * kernel_size;
        std::vector<float> weights(weights_size);
        std::vector<float> bias(out_channels);
        
        file.read(reinterpret_cast<char*>(weights.data()), weights_size * sizeof(float));
        file.read(reinterpret_cast<char*>(bias.data()), out_channels * sizeof(float));
        
        conv1_->load_weights(weights.data(), bias.data());
    }
    
    // Load conv2 weights
    {
        int in_channels, out_channels, kernel_size;
        file.read(reinterpret_cast<char*>(&in_channels), sizeof(in_channels));
        file.read(reinterpret_cast<char*>(&out_channels), sizeof(out_channels));
        file.read(reinterpret_cast<char*>(&kernel_size), sizeof(kernel_size));
        
        if (in_channels != 32 || out_channels != 64 || kernel_size != 3) {
            throw std::runtime_error("Conv2 parameters mismatch");
        }
        
        int weights_size = in_channels * out_channels * kernel_size * kernel_size;
        std::vector<float> weights(weights_size);
        std::vector<float> bias(out_channels);
        
        file.read(reinterpret_cast<char*>(weights.data()), weights_size * sizeof(float));
        file.read(reinterpret_cast<char*>(bias.data()), out_channels * sizeof(float));
        
        conv2_->load_weights(weights.data(), bias.data());
    }
    
    // Load conv3 weights
    {
        int in_channels, out_channels, kernel_size;
        file.read(reinterpret_cast<char*>(&in_channels), sizeof(in_channels));
        file.read(reinterpret_cast<char*>(&out_channels), sizeof(out_channels));
        file.read(reinterpret_cast<char*>(&kernel_size), sizeof(kernel_size));
        
        if (in_channels != 64 || out_channels != 128 || kernel_size != 3) {
            throw std::runtime_error("Conv3 parameters mismatch");
        }
        
        int weights_size = in_channels * out_channels * kernel_size * kernel_size;
        std::vector<float> weights(weights_size);
        std::vector<float> bias(out_channels);
        
        file.read(reinterpret_cast<char*>(weights.data()), weights_size * sizeof(float));
        file.read(reinterpret_cast<char*>(bias.data()), out_channels * sizeof(float));
        
        conv3_->load_weights(weights.data(), bias.data());
    }
    
    // Load final FC (1x1 conv) weights
    {
        int in_channels, out_channels, kernel_size;
        file.read(reinterpret_cast<char*>(&in_channels), sizeof(in_channels));
        file.read(reinterpret_cast<char*>(&out_channels), sizeof(out_channels));
        file.read(reinterpret_cast<char*>(&kernel_size), sizeof(kernel_size));
        
        if (in_channels != 128 || out_channels != num_classes_ || kernel_size != 1) {
            throw std::runtime_error("FC parameters mismatch");
        }
        
        int weights_size = in_channels * out_channels * kernel_size * kernel_size;
        std::vector<float> weights(weights_size);
        std::vector<float> bias(out_channels);
        
        file.read(reinterpret_cast<char*>(weights.data()), weights_size * sizeof(float));
        file.read(reinterpret_cast<char*>(bias.data()), out_channels * sizeof(float));
        
        fc_->load_weights(weights.data(), bias.data());
    }
    
    // Verify end of file
    char dummy;
    if (file.read(&dummy, 1)) {
        std::cerr << "Warning: Weight file contains extra data" << std::endl;
    }
    
    std::cout << "Successfully loaded weights" << std::endl;
}

// Add new function to save a model
bool save_model(const Model& model, const std::string& filename) {
    try {
        model.save(filename);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
        return false;
    }
}

// Add new function to load a model
std::unique_ptr<Model> load_model(const std::string& filename) {
    try {
        auto model = std::make_unique<Model>();
        model->load(filename);
        return model;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return nullptr;
    }
}

// Add tensor utility functions

// Function to concatenate tensors along a dimension
Tensor tensor_concatenate(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) {
        throw std::runtime_error("Cannot concatenate empty tensor list");
    }
    
    // Check compatibility
    const auto& first_shape = tensors[0].shape();
    int num_dims = first_shape.size();
    
    if (dim < 0 || dim >= num_dims) {
        throw std::runtime_error("Invalid concatenation dimension");
    }
    
    // Calculate output shape
    std::vector<int> output_shape = first_shape;
    int concat_dim_size = 0;
    
    for (const auto& tensor : tensors) {
        const auto& shape = tensor.shape();
        
        if (shape.size() != num_dims) {
            throw std::runtime_error("Tensor dimensions mismatch for concatenation");
        }
        
        for (int i = 0; i < num_dims; ++i) {
            if (i != dim && shape[i] != first_shape[i]) {
                throw std::runtime_error("Tensor shapes mismatch for concatenation");
            }
        }
        
        concat_dim_size += shape[dim];
    }
    
    output_shape[dim] = concat_dim_size;
    
    // Create output tensor on CPU
    Tensor output(output_shape, DeviceType::CPU, tensors[0].data_type());
    
    // Copy data
    size_t offset = 0;
    size_t element_size = output.element_size();
    
    // Calculate strides
    std::vector<size_t> strides(num_dims, 1);
    for (int i = num_dims - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * output_shape[i + 1];
    }
    
    // Copy each tensor
    for (const auto& tensor : tensors) {
        // Move to CPU if needed
        Tensor cpu_tensor = tensor;
        if (tensor.device_type() == DeviceType::CUDA) {
            cpu_tensor.to(DeviceType::CPU);
        }
        
        const auto& shape = cpu_tensor.shape();
        
        // Calculate indices and copy data
        std::vector<int> indices(num_dims, 0);
        
        // Number of elements to copy at once (contiguous elements)
        size_t copy_size = element_size;
        for (int i = dim + 1; i < num_dims; ++i) {
            copy_size *= shape[i];
        }
        
        // Number of copies to perform
        size_t num_copies = 1;
        for (int i = 0; i < dim; ++i) {
            num_copies *= shape[i];
        }
        
        // Perform the copies
        char* output_data = static_cast<char*>(output.data());
        const char* input_data = static_cast<const char*>(cpu_tensor.data());
        
        for (size_t i = 0; i < num_copies; ++i) {
            // Calculate source and destination offsets
            size_t src_offset = 0;
            size_t dst_offset = 0;
            
            for (int j = 0; j < dim; ++j) {
                dst_offset += indices[j] * strides[j] * element_size;
            }
            
            dst_offset += offset * strides[dim] * element_size;
            
            // Copy data
            std::memcpy(output_data + dst_offset, input_data + src_offset, copy_size);
            
            // Update indices
            for (int j = 0; j < dim; ++j) {
                indices[j]++;
                if (indices[j] < shape[j]) {
                    break;
                }
                indices[j] = 0;
            }
        }
        
        // Update offset
        offset += shape[dim];
    }
    
    // Move to original device if needed
    if (tensors[0].device_type() == DeviceType::CUDA) {
        output.to(DeviceType::CUDA);
    }
    
    return output;
}

// Function to slice a tensor
Tensor tensor_slice(const Tensor& input, int dim, int start, int end) {
    const auto& shape = input.shape();
    
    if (dim < 0 || dim >= shape.size()) {
        throw std::runtime_error("Invalid slice dimension");
    }
    
    if (start < 0 || end > shape[dim] || start >= end) {
        throw std::runtime_error("Invalid slice range");
    }
    
    // Calculate output shape
    std::vector<int> output_shape = shape;
    output_shape[dim] = end - start;
    
    // Create output tensor
    Tensor output(output_shape, input.device_type(), input.data_type());
    
    // Move to CPU for processing
    Tensor input_cpu = input;
    Tensor output_cpu = output;
    
    if (input.device_type() == DeviceType::CUDA) {
        input_cpu.to(DeviceType::CPU);
        output_cpu.to(DeviceType::CPU);
    }
    
    // Calculate strides
    std::vector<size_t> strides(shape.size(), 1);
    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    
    // Copy data
    size_t element_size = input.element_size();
    char* output_data = static_cast<char*>(output_cpu.data());
    const char* input_data = static_cast<const char*>(input_cpu.data());
    
    // Calculate indices and copy data
    std::vector<int> indices(shape.size(), 0);
    
    // Number of elements before the slice dimension
    size_t pre_dim_elements = 1;
    for (int i = 0; i < dim; ++i) {
        pre_dim_elements *= shape[i];
    }
    
    // Number of elements after the slice dimension
    size_t post_dim_elements = 1;
    for (int i = dim + 1; i < shape.size(); ++i) {
        post_dim_elements *= shape[i];
    }
    
    // Process each element before the dimension
    for (size_t i = 0; i < pre_dim_elements; ++i) {
        // Process each element in the slice
        for (int j = start; j < end; ++j) {
            // Process each element after the dimension
            size_t src_offset = (i * shape[dim] + j) * post_dim_elements * element_size;
            size_t dst_offset = (i * (end - start) + (j - start)) * post_dim_elements * element_size;
            
            std::memcpy(output_data + dst_offset, input_data + src_offset, 
                      post_dim_elements * element_size);
        }
    }
    
    // Move back to original device if needed
    if (input.device_type() == DeviceType::CUDA) {
        output_cpu.to(DeviceType::CUDA);
        output = output_cpu;
    }
    
    return output;
}

} // namespace cnn_cuda
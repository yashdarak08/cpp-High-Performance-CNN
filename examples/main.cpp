#include "cnn_cuda.h"
#include "utils/image_utils.h"
#include "layers/conv_layer.h"
#include "layers/pooling_layer.h"
#include "layers/activation_layer.h"
#include "layers/norm_layer.h"

#include <iostream>
#include <string>
#include <vector>
#include <chrono>

using namespace cnn_cuda;

int main(int argc, char** argv) {
    // Initialize the library
    if (!cnn_cuda::init()) {
        std::cerr << "Failed to initialize CNN CUDA library" << std::endl;
        return -1;
    }
    
    std::cout << "CNN CUDA Library v" << cnn_cuda::VERSION << " initialized successfully" << std::endl;
    
    // Create a simple model and run a dummy tensor through it
    try {
        // Create a 1x3x224x224 random input tensor
        Tensor input({1, 3, 224, 224});
        
        // Initialize with random data
        std::vector<float> random_data(input.size());
        for (int i = 0; i < input.size(); ++i) {
            random_data[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        
        Tensor cpu_input(input.shape(), random_data.data(), DeviceType::CPU);
        input.copy_from(cpu_input);
        
        // Create a simple model
        std::cout << "Creating a simple CNN model..." << std::endl;
        
        ConvLayer conv1(3, 16, 3, 1, 1);
        MaxPoolingLayer pool1(2, 2);
        BatchNormLayer bn1(16);
        
        ConvLayer conv2(16, 32, 3, 1, 1);
        MaxPoolingLayer pool2(2, 2);
        
        // Initialize parameters
        conv1.initialize_parameters();
        conv2.initialize_parameters();
        
        // Measure inference time
        std::cout << "Running inference..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        // Forward pass
        Tensor out = conv1.forward(input);
        out = pool1.forward(out);
        out = bn1.forward(out);
        out = relu(out);
        
        out = conv2.forward(out);
        out = pool2.forward(out);
        out = relu(out);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::cout << "Inference completed in " << duration << " ms" << std::endl;
        std::cout << "Output tensor shape: [";
        for (size_t i = 0; i < out.shape().size(); ++i) {
            std::cout << out.shape()[i];
            if (i < out.shape().size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Memory usage statistics
        size_t device_mem = MemoryPool::instance().get_device_memory_usage();
        size_t host_mem = MemoryPool::instance().get_host_memory_usage();
        
        std::cout << "Memory usage - Device: " << device_mem / (1024 * 1024) << " MB, "
                  << "Host: " << host_mem / (1024 * 1024) << " MB" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    // Clean up
    cnn_cuda::shutdown();
    std::cout << "CNN CUDA library shut down successfully" << std::endl;
    
    return 0;
}
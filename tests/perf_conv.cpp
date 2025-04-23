#include "cnn_cuda.h"
#include "layers/conv_layer.h"
#include "test_utils.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

using namespace cnn_cuda;
using namespace cnn_cuda::test;

// Benchmark different convolution configurations
void benchmark_convolutions() {
    TestDataGenerator generator(42);
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n===== Convolution Performance Benchmark =====\n" << std::endl;
    
    // Configurations to test
    struct ConvConfig {
        int batch_size;
        int in_channels;
        int out_channels;
        int input_size;
        int kernel_size;
        int stride;
        int padding;
        std::string name;
    };
    
    std::vector<ConvConfig> configs = {
        {1, 3, 64, 224, 7, 2, 3, "ResNet-50 First Layer"},
        {1, 64, 64, 56, 3, 1, 1, "ResNet-50 Stage 2 (1x1)"},
        {1, 128, 256, 28, 3, 2, 1, "ResNet-50 Downsampling"},
        {32, 3, 32, 32, 3, 1, 1, "Small CNN Batch"},
        {1, 512, 512, 7, 3, 1, 1, "Dense Features"}
    };
    
    const int num_runs = 100;  // Number of runs for each configuration
    const int warmup_runs = 10; // Warmup runs
    
    // Header
    std::cout << std::setw(25) << "Configuration" 
              << std::setw(15) << "Direct (ms)" 
              << std::setw(15) << "Im2Col (ms)" 
              << std::setw(15) << "Winograd (ms)" 
              << std::setw(15) << "Best (ms)" << std::endl;
    std::cout << std::string(85, '-') << std::endl;
    
    for (const auto& config : configs) {
        // Create input tensor
        std::vector<int> input_shape = {config.batch_size, config.in_channels, 
                                       config.input_size, config.input_size};
        Tensor input = generator.generateRandomTensor(input_shape, DeviceType::CUDA);
        
        // Create layers for different implementations
        ConvLayer direct_conv(config.in_channels, config.out_channels, 
                            config.kernel_size, config.stride, config.padding);
        ConvLayer im2col_conv(config.in_channels, config.out_channels, 
                             config.kernel_size, config.stride, config.padding);
        ConvLayer winograd_conv(config.in_channels, config.out_channels, 
                               config.kernel_size, config.stride, config.padding);
        
        // Initialize parameters
        direct_conv.initialize_parameters();
        im2col_conv.initialize_parameters();
        winograd_conv.initialize_parameters();
        
        // Create output tensors
        int out_size = (config.input_size + 2*config.padding - config.kernel_size) / config.stride + 1;
        std::vector<int> output_shape = {config.batch_size, config.out_channels, out_size, out_size};
        
        Tensor output_direct(output_shape, DeviceType::CUDA);
        Tensor output_im2col(output_shape, DeviceType::CUDA);
        Tensor output_winograd(output_shape, DeviceType::CUDA);
        
        // Warmup
        for (int i = 0; i < warmup_runs; ++i) {
            direct_conv.forward(input);
            im2col_conv.forward(input);
            if (config.kernel_size == 3 && config.stride == 1) {
                winograd_conv.forward(input);
            }
        }
        
        // Benchmark direct convolution
        Timer timer;
        for (int i = 0; i < num_runs; ++i) {
            output_direct = direct_conv.forward(input);
        }
        double direct_time = timer.elapsedMilliseconds() / num_runs;
        
        // Benchmark im2col convolution
        timer.reset();
        for (int i = 0; i < num_runs; ++i) {
            output_im2col = im2col_conv.forward(input);
        }
        double im2col_time = timer.elapsedMilliseconds() / num_runs;
        
        // Benchmark Winograd convolution (only for 3x3 with stride 1)
        double winograd_time = 0.0;
        if (config.kernel_size == 3 && config.stride == 1) {
            timer.reset();
            for (int i = 0; i < num_runs; ++i) {
                output_winograd = winograd_conv.forward(input);
            }
            winograd_time = timer.elapsedMilliseconds() / num_runs;
        } else {
            winograd_time = std::numeric_limits<double>::infinity();
        }
        
        // Find the best time
        double best_time = std::min({direct_time, im2col_time, winograd_time});
        
        // Print results
        std::cout << std::setw(25) << config.name
                  << std::setw(15) << direct_time
                  << std::setw(15) << im2col_time;
        
        if (config.kernel_size == 3 && config.stride == 1) {
            std::cout << std::setw(15) << winograd_time;
        } else {
            std::cout << std::setw(15) << "N/A";
        }
        
        std::cout << std::setw(15) << best_time << std::endl;
    }
}

int main() {
    if (!cnn_cuda::init()) {
        std::cerr << "Failed to initialize CNN CUDA library" << std::endl;
        return -1;
    }
    
    try {
        benchmark_convolutions();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    cnn_cuda::shutdown();
    return 0;
}
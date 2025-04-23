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

// Simple CNN model for image classification
class SimpleCNN {
public:
    SimpleCNN(int num_classes) : num_classes_(num_classes) {
        // Define layers for a simple CNN
        conv1_ = std::make_unique<ConvLayer>(3, 32, 3, 1, 1);
        pool1_ = std::make_unique<MaxPoolingLayer>(2, 2);
        conv2_ = std::make_unique<ConvLayer>(32, 64, 3, 1, 1);
        pool2_ = std::make_unique<MaxPoolingLayer>(2, 2);
        conv3_ = std::make_unique<ConvLayer>(64, 128, 3, 1, 1);
        pool3_ = std::make_unique<MaxPoolingLayer>(2, 2);
        fc_ = std::make_unique<ConvLayer>(128, num_classes, 1);  // 1x1 conv as FC layer
        
        // Initialize parameters
        conv1_->initialize_parameters();
        conv2_->initialize_parameters();
        conv3_->initialize_parameters();
        fc_->initialize_parameters();
    }
    
    // Forward pass
    Tensor forward(const Tensor& x) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Feature extraction
        Tensor out = conv1_->forward(x);
        out = pool1_->forward(out);
        out = relu(out);
        
        out = conv2_->forward(out);
        out = pool2_->forward(out);
        out = relu(out);
        
        out = conv3_->forward(out);
        out = pool3_->forward(out);
        out = relu(out);
        
        // Global average pooling
        std::vector<int> shape = out.shape();
        int batch_size = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];
        
        Tensor gap = global_avg_pool(out);
        
        // Classification head (1x1 convolution instead of FC layer)
        Tensor logits = fc_->forward(gap);
        
        // Reshape to [batch_size, num_classes]
        logits.reshape({batch_size, num_classes_, 1, 1});
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Forward pass completed in " << duration << " ms" << std::endl;
        
        return logits;
    }
    
    // Load weights for all layers
    void load_weights(const std::string& weights_file) {
        // Implementation to load weights from file
        // Omitted for brevity
    }
    
private:
    int num_classes_;
    
    std::unique_ptr<ConvLayer> conv1_;
    std::unique_ptr<MaxPoolingLayer> pool1_;
    std::unique_ptr<ConvLayer> conv2_;
    std::unique_ptr<MaxPoolingLayer> pool2_;
    std::unique_ptr<ConvLayer> conv3_;
    std::unique_ptr<MaxPoolingLayer> pool3_;
    std::unique_ptr<ConvLayer> fc_;
    
    // Helper functions
    Tensor relu(const Tensor& x) {
        Tensor output(x.shape());
        tensor_relu(x, output);
        return output;
    }
    
    Tensor global_avg_pool(const Tensor& x) {
        // Get input shape
        std::vector<int> shape = x.shape();
        int batch_size = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];
        
        // Create output tensor [batch_size, channels, 1, 1]
        Tensor output({batch_size, channels, 1, 1});
        
        // Forward declaration of the global avg pool CUDA function
        extern void launch_global_avg_pool(
            const float* input,
            float* output,
            int batch_size,
            int channels,
            int height,
            int width,
            cudaStream_t stream);
        
        // Launch kernel
        launch_global_avg_pool(
            x.data(),
            output.data(),
            batch_size,
            channels,
            height,
            width,
            0 // Default stream
        );
        
        return output;
    }
};

int main(int argc, char** argv) {
    // Parse command line arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }
    
    std::string image_path = argv[1];
    
    try {
        // Load and preprocess image
        Tensor image = ImageUtils::load_image(image_path, 224, 224);
        
        // Normalize with ImageNet stats
        std::vector<float> mean = {0.485f, 0.456f, 0.406f};
        std::vector<float> std = {0.229f, 0.224f, 0.225f};
        image = ImageUtils::normalize(image, mean, std);
        
        // Add batch dimension [1, 3, 224, 224]
        image.reshape({1, 3, 224, 224});
                
        // Create model (10 classes for this example)
        SimpleCNN model(10);

        // Optional: load pre-trained weights
        // model.load_weights("model_weights.bin");

        // Run inference
        Tensor predictions = model.forward(image);

        // Get top prediction
        // Implementation omitted for brevity

        std::cout << "Classification completed successfully" << std::endl;

        } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
        }

        // Get top prediction
        Tensor logits = predictions.reshape({batch_size, num_classes});

        // Move to CPU for processing
        logits.to(DeviceType::CPU);
        float* logits_data = logits.data();

        // Find the class with highest score
        int top_class = 0;
        float top_score = logits_data[0];

        for (int i = 1; i < num_classes; ++i) {
            if (logits_data[i] > top_score) {
                top_score = logits_data[i];
                top_class = i;
            }
        }

        // Sample class names for demonstration (would be loaded from a file in practice)
        std::vector<std::string> class_names = {
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        };

        // Print result
        std::cout << "Top prediction: " << class_names[top_class] << " with confidence " << top_score << std::endl;

        return 0;
        }
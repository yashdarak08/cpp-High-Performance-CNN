#include "utils/image_utils.h"
#include <stdexcept>
#include <algorithm>

// For simplicity, using stb_image for image loading
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace cnn_cuda {

Tensor ImageUtils::load_image(const std::string& file_path, int target_width, int target_height) {
    int width, height, channels;
    unsigned char* data = stbi_load(file_path.c_str(), &width, &height, &channels, 3);
    
    if (!data) {
        throw std::runtime_error("Failed to load image: " + file_path);
    }
    
    // Convert to float, scale to [0, 1], and change from HWC to CHW layout
    std::vector<float> float_data(3 * width * height);
    
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < 3; ++c) {
                float_data[c * width * height + h * width + w] = 
                    static_cast<float>(data[(h * width + w) * 3 + c]) / 255.0f;
            }
        }
    }
    
    // Free stb_image data
    stbi_image_free(data);
    
    // Create tensor on CPU
    Tensor image({3, height, width}, float_data.data(), DeviceType::CPU);
    
    // Resize if needed
    if (target_width > 0 && target_height > 0 && (width != target_width || height != target_height)) {
        image = resize(image, target_width, target_height);
    }
    
    // Move to GPU
    image.to(DeviceType::CUDA);
    
    return image;
}

bool ImageUtils::save_image(const Tensor& tensor, const std::string& file_path) {
    const auto& shape = tensor.shape();
    
    if (shape.size() != 3 || shape[0] != 3) {
        throw std::runtime_error("Expected 3D tensor with 3 channels for image saving");
    }
    
    int channels = shape[0];
    int height = shape[1];
    int width = shape[2];
    
    // Create CPU tensor if needed
    Tensor cpu_tensor = tensor;
    if (tensor.device_type() == DeviceType::CUDA) {
        cpu_tensor.to(DeviceType::CPU);
    }
    
    // Convert from float [0, 1] to byte [0, 255] and from CHW to HWC
    std::vector<unsigned char> byte_data(width * height * channels);
    
    const float* float_data = cpu_tensor.data();
    
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                float val = float_data[c * width * height + h * width + w];
                val = std::min(std::max(val, 0.0f), 1.0f); // Clamp to [0, 1]
                byte_data[(h * width + w) * channels + c] = static_cast<unsigned char>(val * 255.0f);
            }
        }
    }
    
    // Save image
    int result = stbi_write_png(file_path.c_str(), width, height, channels, byte_data.data(), width * channels);
    
    return result != 0;
}

Tensor ImageUtils::normalize(const Tensor& image, const std::vector<float>& mean, const std::vector<float>& std) {
    // Implementation omitted for brevity
    // This would involve a CUDA kernel to normalize each pixel
    return image;
}

Tensor ImageUtils::resize(const Tensor& image, int target_width, int target_height) {
    // Implementation omitted for brevity
    // This would involve a CUDA kernel for resizing
    return image;
}

Tensor ImageUtils::center_crop(const Tensor& image, int crop_width, int crop_height) {
    // Implementation omitted for brevity
    // This would involve a CUDA kernel or simple tensor slice operation
    return image;
}

Tensor ImageUtils::batch_images(const std::vector<Tensor>& images) {
    if (images.empty()) {
        throw std::runtime_error("Cannot batch empty image list");
    }
    
    // Get shapes from first image
    const auto& first_shape = images[0].shape();
    
    if (first_shape.size() != 3) {
        throw std::runtime_error("Expected 3D tensors for batching");
    }
    
    int channels = first_shape[0];
    int height = first_shape[1];
    int width = first_shape[2];
    int batch_size = images.size();
    
    // Create output tensor
    Tensor batch({batch_size, channels, height, width});
    
    // Copy each image into the batch
    for (int i = 0; i < batch_size; ++i) {
        const auto& img = images[i];
        
        // Check shape
        const auto& img_shape = img.shape();
        if (img_shape[0] != channels || img_shape[1] != height || img_shape[2] != width) {
            throw std::runtime_error("All images must have the same dimensions for batching");
        }
        
        // Copy data (this is a simplified version, real implementation would use CUDA kernel)
        // For each image in batch:
        //   - Calculate offset in batch tensor
        //   - Copy image data to this offset
    }
    
    return batch;
}

} // namespace cnn_cuda
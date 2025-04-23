#include "utils/image_utils.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

// Define external library dependencies if needed (like stb_image)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace cnn_cuda {

// Load an image from file
Tensor ImageUtils::load_image(const std::string& file_path, int target_width, int target_height) {
    int width, height, channels;
    unsigned char* data = stbi_load(file_path.c_str(), &width, &height, &channels, 3);
    
    if (!data) {
        throw std::runtime_error("Failed to load image: " + file_path);
    }
    
    // Convert to float and normalize to [0, 1]
    size_t size = width * height * 3;
    std::vector<float> float_data(size);
    
    for (size_t i = 0; i < size; ++i) {
        float_data[i] = static_cast<float>(data[i]) / 255.0f;
    }
    
    stbi_image_free(data);
    
    // Create tensor with shape [3, height, width] (CHW format)
    Tensor image({3, height, width}, float_data.data(), DeviceType::CPU);
    
    // Resize if target dimensions are specified
    if (target_width > 0 && target_height > 0 && (width != target_width || height != target_height)) {
        image = resize(image, target_width, target_height);
    }
    
    return image;
}

// Save tensor as an image
bool ImageUtils::save_image(const Tensor& tensor, const std::string& file_path) {
    const auto& shape = tensor.shape();
    
    if (shape.size() != 3 || shape[0] != 3) {
        throw std::runtime_error("Expected 3D tensor with 3 channels for saving as image");
    }
    
    int channels = shape[0];
    int height = shape[1];
    int width = shape[2];
    
    // Create a CPU tensor for the work
    Tensor cpu_tensor = tensor;
    if (tensor.device_type() == DeviceType::CUDA) {
        cpu_tensor.to(DeviceType::CPU);
    }
    
    // Convert to unsigned char
    std::vector<unsigned char> image_data(width * height * channels);
    
    const float* tensor_data = cpu_tensor.data();
    
    // Transpose from CHW to HWC format and scale to [0, 255]
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                float pixel = tensor_data[c * height * width + h * width + w];
                // Clamp to [0, 1]
                pixel = std::max(0.0f, std::min(1.0f, pixel));
                // Scale to [0, 255]
                unsigned char value = static_cast<unsigned char>(pixel * 255.0f);
                // Store in HWC format
                image_data[(h * width + w) * channels + c] = value;
            }
        }
    }
    
    // Save the image
    int result = stbi_write_png(file_path.c_str(), width, height, channels, 
                              image_data.data(), width * channels);
    
    return result != 0;
}

// Normalize method (already implemented)
Tensor ImageUtils::normalize(const Tensor& image, const std::vector<float>& mean, const std::vector<float>& std) {
    const auto& shape = image.shape();
    
    if (shape.size() != 3 || shape[0] != 3) {
        throw std::runtime_error("Expected 3D tensor with 3 channels for normalization");
    }
    
    int channels = shape[0];
    int height = shape[1];
    int width = shape[2];
    
    if (mean.size() != channels || std.size() != channels) {
        throw std::runtime_error("Mean and std vectors must have same size as number of channels");
    }
    
    // Create a CPU tensor for the work
    Tensor cpu_image = image;
    if (image.device_type() == DeviceType::CUDA) {
        cpu_image.to(DeviceType::CPU);
    }
    
    // Create output tensor on CPU
    Tensor output(shape, nullptr, DeviceType::CPU);
    
    const float* input_data = cpu_image.data();
    float* output_data = output.data();
    
    // Normalize each channel
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int idx = c * height * width + h * width + w;
                output_data[idx] = (input_data[idx] - mean[c]) / std[c];
            }
        }
    }
    
    // Move result to match original device type
    if (image.device_type() == DeviceType::CUDA) {
        output.to(DeviceType::CUDA);
    }
    
    return output;
}

// Simple bilinear interpolation for a single pixel and channel
inline float bilinear_interpolate(
    const float* data,
    int height, int width,
    float y, float x)
{
    // Get the 4 nearest pixels
    int x1 = static_cast<int>(std::floor(x));
    int y1 = static_cast<int>(std::floor(y));
    int x2 = std::min(x1 + 1, width - 1);
    int y2 = std::min(y1 + 1, height - 1);
    
    // Calculate interpolation weights
    float wx = x - x1;
    float wy = y - y1;
    
    // Perform bilinear interpolation
    float val = (1.0f - wy) * (1.0f - wx) * data[y1 * width + x1] +
                wy * (1.0f - wx) * data[y2 * width + x1] +
                (1.0f - wy) * wx * data[y1 * width + x2] +
                wy * wx * data[y2 * width + x2];
    
    return val;
}

// Resize method (already implemented)
Tensor ImageUtils::resize(const Tensor& image, int target_width, int target_height) {
    const auto& shape = image.shape();
    
    if (shape.size() != 3) {
        throw std::runtime_error("Expected 3D tensor for resize");
    }
    
    int channels = shape[0];
    int height = shape[1];
    int width = shape[2];
    
    // Create a CPU tensor for the work
    Tensor cpu_image = image;
    if (image.device_type() == DeviceType::CUDA) {
        cpu_image.to(DeviceType::CPU);
    }
    
    // Create output tensor on CPU
    Tensor output({channels, target_height, target_width}, nullptr, DeviceType::CPU);
    
    const float* input_data = cpu_image.data();
    float* output_data = output.data();
    
    // Scale factors
    float scale_y = static_cast<float>(height) / target_height;
    float scale_x = static_cast<float>(width) / target_width;
    
    // Resize using bilinear interpolation
    for (int c = 0; c < channels; ++c) {
        const float* channel_input = input_data + c * height * width;
        float* channel_output = output_data + c * target_height * target_width;
        
        for (int y = 0; y < target_height; ++y) {
            for (int x = 0; x < target_width; ++x) {
                float src_y = (y + 0.5f) * scale_y - 0.5f;
                float src_x = (x + 0.5f) * scale_x - 0.5f;
                
                // Clamp coordinates
                src_y = std::max(0.0f, std::min(src_y, static_cast<float>(height - 1)));
                src_x = std::max(0.0f, std::min(src_x, static_cast<float>(width - 1)));
                
                // Bilinear interpolation
                channel_output[y * target_width + x] = bilinear_interpolate(
                    channel_input, height, width, src_y, src_x);
            }
        }
    }
    
    // Move result to match original device type
    if (image.device_type() == DeviceType::CUDA) {
        output.to(DeviceType::CUDA);
    }
    
    return output;
}

// Center crop method (already implemented)
Tensor ImageUtils::center_crop(const Tensor& image, int crop_width, int crop_height) {
    const auto& shape = image.shape();
    
    if (shape.size() != 3) {
        throw std::runtime_error("Expected 3D tensor for center crop");
    }
    
    int channels = shape[0];
    int height = shape[1];
    int width = shape[2];
    
    if (crop_width > width || crop_height > height) {
        throw std::runtime_error("Crop dimensions cannot be larger than image dimensions");
    }
    
    // Calculate crop coordinates
    int start_h = (height - crop_height) / 2;
    int start_w = (width - crop_width) / 2;
    
    // Create a CPU tensor for the work
    Tensor cpu_image = image;
    if (image.device_type() == DeviceType::CUDA) {
        cpu_image.to(DeviceType::CPU);
    }
    
    // Create output tensor on CPU
    Tensor output({channels, crop_height, crop_width}, nullptr, DeviceType::CPU);
    
    const float* input_data = cpu_image.data();
    float* output_data = output.data();
    
    // Perform center crop
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_height; ++h) {
            for (int w = 0; w < crop_width; ++w) {
                int in_idx = c * height * width + (start_h + h) * width + (start_w + w);
                int out_idx = c * crop_height * crop_width + h * crop_width + w;
                output_data[out_idx] = input_data[in_idx];
            }
        }
    }
    
    // Move result to match original device type
    if (image.device_type() == DeviceType::CUDA) {
        output.to(DeviceType::CUDA);
    }
    
    return output;
}

// Batch images method (missing implementation)
Tensor ImageUtils::batch_images(const std::vector<Tensor>& images) {
    if (images.empty()) {
        throw std::runtime_error("Cannot batch empty image list");
    }
    
    // Check that all images have the same shape
    const auto& first_shape = images[0].shape();
    
    if (first_shape.size() != 3) {
        throw std::runtime_error("Expected 3D tensors for batching");
    }
    
    int channels = first_shape[0];
    int height = first_shape[1];
    int width = first_shape[2];
    
    for (size_t i = 1; i < images.size(); ++i) {
        const auto& shape = images[i].shape();
        
        if (shape.size() != 3 || 
            shape[0] != channels || 
            shape[1] != height || 
            shape[2] != width) {
            throw std::runtime_error("All images must have the same dimensions for batching");
        }
    }
    
    // Create the output tensor with batch dimension
    int batch_size = static_cast<int>(images.size());
    Tensor output({batch_size, channels, height, width}, nullptr, DeviceType::CPU);
    
    float* output_data = output.data();
    int image_size = channels * height * width;
    
    // Copy each image to the batched tensor
    for (int b = 0; b < batch_size; ++b) {
        Tensor cpu_image = images[b];
        if (images[b].device_type() == DeviceType::CUDA) {
            cpu_image.to(DeviceType::CPU);
        }
        
        const float* image_data = cpu_image.data();
        float* batch_ptr = output_data + b * image_size;
        
        std::memcpy(batch_ptr, image_data, image_size * sizeof(float));
    }
    
    // If all input images were on GPU, move the result to GPU as well
    bool all_gpu = true;
    for (const auto& img : images) {
        if (img.device_type() != DeviceType::CUDA) {
            all_gpu = false;
            break;
        }
    }
    
    if (all_gpu) {
        output.to(DeviceType::CUDA);
    }
    
    return output;
}

} // namespace cnn_cuda
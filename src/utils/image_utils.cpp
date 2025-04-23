// Implement normalize method
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

// Implement resize method
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

// Implement center_crop method
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
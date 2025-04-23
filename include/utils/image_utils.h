#pragma once

#include <string>
#include <vector>
#include "../tensor.h"

namespace cnn_cuda {

class ImageUtils {
public:
    // Load an image from file
    static Tensor load_image(const std::string& file_path, int target_width = -1, int target_height = -1);
    
    // Save tensor as an image
    static bool save_image(const Tensor& tensor, const std::string& file_path);
    
    // Preprocessing methods
    static Tensor normalize(const Tensor& image, const std::vector<float>& mean, const std::vector<float>& std);
    static Tensor resize(const Tensor& image, int target_width, int target_height);
    static Tensor center_crop(const Tensor& image, int crop_width, int crop_height);
    
    // Batch processing
    static Tensor batch_images(const std::vector<Tensor>& images);
};

} // namespace cnn_cuda
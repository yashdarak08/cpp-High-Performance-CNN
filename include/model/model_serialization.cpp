#include "model/model_serialization.h"
#include <fstream>
#include <stdexcept>
#include <vector>

namespace cnn_cuda {

// Model methods
void Model::addLayer(std::unique_ptr<SerializableLayer> layer) {
    layers_.push_back(std::move(layer));
}

Tensor Model::forward(const Tensor& input) {
    if (layers_.empty()) {
        throw std::runtime_error("Model has no layers");
    }
    
    Tensor current = input;
    for (const auto& layer : layers_) {
        current = layer->forward(current);
    }
    
    return current;
}

void Model::save(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    
    // Write number of layers
    size_t num_layers = layers_.size();
    out.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    
    // Write each layer
    for (const auto& layer : layers_) {
        // Write layer type
        LayerType type = layer->getType();
        out.write(reinterpret_cast<const char*>(&type), sizeof(type));
        
        // Write layer data
        layer->serialize(out);
    }
}

void Model::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }
    
    // Clear existing layers
    layers_.clear();
    
    // Read number of layers
    size_t num_layers;
    in.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    
    // Read each layer
    for (size_t i = 0; i < num_layers; ++i) {
        // Read layer type
        LayerType type;
        in.read(reinterpret_cast<char*>(&type), sizeof(type));
        
        // Create layer
        auto layer = createLayer(type);
        
        // Read layer data
        layer->deserialize(in);
        
        // Add layer to model
        layers_.push_back(std::move(layer));
    }
}

size_t Model::numLayers() const {
    return layers_.size();
}

SerializableLayer* Model::getLayer(size_t index) {
    if (index >= layers_.size()) {
        throw std::out_of_range("Layer index out of range");
    }
    return layers_[index].get();
}

// SerializableConvLayer methods
SerializableConvLayer::SerializableConvLayer(int in_channels, int out_channels, 
                                          int kernel_size, int stride, int padding)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding) {
    
    conv_layer_ = std::make_unique<ConvLayer>(in_channels_, out_channels_, 
                                             kernel_size_, stride_, padding_);
    conv_layer_->initialize_parameters();
}

SerializableConvLayer::SerializableConvLayer()
    : in_channels_(0),
      out_channels_(0),
      kernel_size_(0),
      stride_(0),
      padding_(0) {
}

void SerializableConvLayer::serialize(std::ostream& out) const {
    // Write layer parameters
    out.write(reinterpret_cast<const char*>(&in_channels_), sizeof(in_channels_));
    out.write(reinterpret_cast<const char*>(&out_channels_), sizeof(out_channels_));
    out.write(reinterpret_cast<const char*>(&kernel_size_), sizeof(kernel_size_));
    out.write(reinterpret_cast<const char*>(&stride_), sizeof(stride_));
    out.write(reinterpret_cast<const char*>(&padding_), sizeof(padding_));
    
    // Get weights and biases
    const Tensor& weights = conv_layer_->get_weights();
    const Tensor& bias = conv_layer_->get_bias();
    
    // Create CPU copies
    Tensor weights_cpu = weights;
    Tensor bias_cpu = bias;
    
    if (weights.device_type() == DeviceType::CUDA) {
        weights_cpu.to(DeviceType::CPU);
    }
    
    if (bias.device_type() == DeviceType::CUDA) {
        bias_cpu.to(DeviceType::CPU);
    }
    
    // Write weights
    const float* weights_data = weights_cpu.data();
    int weights_size = weights_cpu.size();
    out.write(reinterpret_cast<const char*>(&weights_size), sizeof(weights_size));
    out.write(reinterpret_cast<const char*>(weights_data), weights_size * sizeof(float));
    
    // Write bias
    const float* bias_data = bias_cpu.data();
    int bias_size = bias_cpu.size();
    out.write(reinterpret_cast<const char*>(&bias_size), sizeof(bias_size));
    out.write(reinterpret_cast<const char*>(bias_data), bias_size * sizeof(float));
}

void SerializableConvLayer::deserialize(std::istream& in) {
    // Read layer parameters
    in.read(reinterpret_cast<char*>(&in_channels_), sizeof(in_channels_));
    in.read(reinterpret_cast<char*>(&out_channels_), sizeof(out_channels_));
    in.read(reinterpret_cast<char*>(&kernel_size_), sizeof(kernel_size_));
    in.read(reinterpret_cast<char*>(&stride_), sizeof(stride_));
    in.read(reinterpret_cast<char*>(&padding_), sizeof(padding_));
    
    // Create layer
    conv_layer_ = std::make_unique<ConvLayer>(in_channels_, out_channels_, 
                                             kernel_size_, stride_, padding_);
    
    // Read weights
    int weights_size;
    in.read(reinterpret_cast<char*>(&weights_size), sizeof(weights_size));
    std::vector<float> weights_data(weights_size);
    in.read(reinterpret_cast<char*>(weights_data.data()), weights_size * sizeof(float));
    
    // Read bias
    int bias_size;
    in.read(reinterpret_cast<char*>(&bias_size), sizeof(bias_size));
    std::vector<float> bias_data(bias_size);
    in.read(reinterpret_cast<char*>(bias_data.data()), bias_size * sizeof(float));
    
    // Load weights and bias
    conv_layer_->load_weights(weights_data.data(), bias_data.data());
}

Tensor SerializableConvLayer::forward(const Tensor& input) {
    return conv_layer_->forward(input);
}

void SerializableConvLayer::loadWeights(const float* weights, const float* bias) {
    conv_layer_->load_weights(weights, bias);
}

// SerializableMaxPoolingLayer methods
SerializableMaxPoolingLayer::SerializableMaxPoolingLayer(int kernel_size, int stride)
    : kernel_size_(kernel_size), stride_(stride) {
    
    pool_layer_ = std::make_unique<MaxPoolingLayer>(kernel_size_, stride_);
}

SerializableMaxPoolingLayer::SerializableMaxPoolingLayer()
    : kernel_size_(0), stride_(0) {
}

void SerializableMaxPoolingLayer::serialize(std::ostream& out) const {
    // Write layer parameters
    out.write(reinterpret_cast<const char*>(&kernel_size_), sizeof(kernel_size_));
    out.write(reinterpret_cast<const char*>(&stride_), sizeof(stride_));
}

void SerializableMaxPoolingLayer::deserialize(std::istream& in) {
    // Read layer parameters
    in.read(reinterpret_cast<char*>(&kernel_size_), sizeof(kernel_size_));
    in.read(reinterpret_cast<char*>(&stride_), sizeof(stride_));
    
    // Create layer
    pool_layer_ = std::make_unique<MaxPoolingLayer>(kernel_size_, stride_);
}

Tensor SerializableMaxPoolingLayer::forward(const Tensor& input) {
    return pool_layer_->forward(input);
}

// SerializableAvgPoolingLayer methods
SerializableAvgPoolingLayer::SerializableAvgPoolingLayer(int kernel_size, int stride)
    : kernel_size_(kernel_size), stride_(stride) {
    
    pool_layer_ = std::make_unique<AvgPoolingLayer>(kernel_size_, stride_);
}

SerializableAvgPoolingLayer::SerializableAvgPoolingLayer()
    : kernel_size_(0), stride_(0) {
}

void SerializableAvgPoolingLayer::serialize(std::ostream& out) const {
    // Write layer parameters
    out.write(reinterpret_cast<const char*>(&kernel_size_), sizeof(kernel_size_));
    out.write(reinterpret_cast<const char*>(&stride_), sizeof(stride_));
}

void SerializableAvgPoolingLayer::deserialize(std::istream& in) {
    // Read layer parameters
    in.read(reinterpret_cast<char*>(&kernel_size_), sizeof(kernel_size_));
    in.read(reinterpret_cast<char*>(&stride_), sizeof(stride_));
    
    // Create layer
    pool_layer_ = std::make_unique<AvgPoolingLayer>(kernel_size_, stride_);
}

Tensor SerializableAvgPoolingLayer::forward(const Tensor& input) {
    return pool_layer_->forward(input);
}

// SerializableActivationLayer methods
SerializableActivationLayer::SerializableActivationLayer(ActivationType type, float alpha)
    : type_(type), alpha_(alpha) {
    
    activation_layer_ = std::make_unique<ActivationLayer>(type_, alpha_);
}

SerializableActivationLayer::SerializableActivationLayer()
    : type_(ActivationType::RELU), alpha_(0.01f) {
}

void SerializableActivationLayer::deserialize(std::istream& in) {
    // Read layer parameters
    in.read(reinterpret_cast<char*>(&type_), sizeof(type_));
    in.read(reinterpret_cast<char*>(&alpha_), sizeof(alpha_));
    
    // Create layer
    activation_layer_ = std::make_unique<ActivationLayer>(type_, alpha_);
}

Tensor SerializableActivationLayer::forward(const Tensor& input) {
    return activation_layer_->forward(input);
}

// SerializableBatchNormLayer methods
SerializableBatchNormLayer::SerializableBatchNormLayer(int num_features, float eps, float momentum)
    : num_features_(num_features), eps_(eps), momentum_(momentum) {
    
    bn_layer_ = std::make_unique<BatchNormLayer>(num_features_, eps_, momentum_);
}

SerializableBatchNormLayer::SerializableBatchNormLayer()
    : num_features_(0), eps_(1e-5f), momentum_(0.1f) {
}

void SerializableBatchNormLayer::serialize(std::ostream& out) const {
    // Write layer parameters
    out.write(reinterpret_cast<const char*>(&num_features_), sizeof(num_features_));
    out.write(reinterpret_cast<const char*>(&eps_), sizeof(eps_));
    out.write(reinterpret_cast<const char*>(&momentum_), sizeof(momentum_));
    
    // Get parameters
    const Tensor& gamma = bn_layer_->get_gamma();
    const Tensor& beta = bn_layer_->get_beta();
    const Tensor& running_mean = bn_layer_->get_running_mean();
    const Tensor& running_var = bn_layer_->get_running_var();
    
    // Create CPU copies
    Tensor gamma_cpu = gamma;
    Tensor beta_cpu = beta;
    Tensor running_mean_cpu = running_mean;
    Tensor running_var_cpu = running_var;
    
    if (gamma.device_type() == DeviceType::CUDA) gamma_cpu.to(DeviceType::CPU);
    if (beta.device_type() == DeviceType::CUDA) beta_cpu.to(DeviceType::CPU);
    if (running_mean.device_type() == DeviceType::CUDA) running_mean_cpu.to(DeviceType::CPU);
    if (running_var.device_type() == DeviceType::CUDA) running_var_cpu.to(DeviceType::CPU);
    
    // Write parameters
    int param_size = num_features_;
    
    out.write(reinterpret_cast<const char*>(gamma_cpu.data()), param_size * sizeof(float));
    out.write(reinterpret_cast<const char*>(beta_cpu.data()), param_size * sizeof(float));
    out.write(reinterpret_cast<const char*>(running_mean_cpu.data()), param_size * sizeof(float));
    out.write(reinterpret_cast<const char*>(running_var_cpu.data()), param_size * sizeof(float));
}

void SerializableBatchNormLayer::deserialize(std::istream& in) {
    // Read layer parameters
    in.read(reinterpret_cast<char*>(&num_features_), sizeof(num_features_));
    in.read(reinterpret_cast<char*>(&eps_), sizeof(eps_));
    in.read(reinterpret_cast<char*>(&momentum_), sizeof(momentum_));
    
    // Create layer
    bn_layer_ = std::make_unique<BatchNormLayer>(num_features_, eps_, momentum_);
    
    // Read parameters
    int param_size = num_features_;
    
    std::vector<float> gamma_data(param_size);
    std::vector<float> beta_data(param_size);
    std::vector<float> running_mean_data(param_size);
    std::vector<float> running_var_data(param_size);
    
    in.read(reinterpret_cast<char*>(gamma_data.data()), param_size * sizeof(float));
    in.read(reinterpret_cast<char*>(beta_data.data()), param_size * sizeof(float));
    in.read(reinterpret_cast<char*>(running_mean_data.data()), param_size * sizeof(float));
    in.read(reinterpret_cast<char*>(running_var_data.data()), param_size * sizeof(float));
    
    // Load parameters
    bn_layer_->load_parameters(gamma_data.data(), beta_data.data(), 
                              running_mean_data.data(), running_var_data.data());
}

Tensor SerializableBatchNormLayer::forward(const Tensor& input) {
    return bn_layer_->forward(input);
}

void SerializableBatchNormLayer::loadParameters(const float* gamma, const float* beta, 
                                             const float* running_mean, const float* running_var) {
    bn_layer_->load_parameters(gamma, beta, running_mean, running_var);
}

// Factory function
std::unique_ptr<SerializableLayer> createLayer(LayerType type) {
    switch (type) {
        case LayerType::CONV:
            return std::make_unique<SerializableConvLayer>();
        case LayerType::MAX_POOLING:
            return std::make_unique<SerializableMaxPoolingLayer>();
        case LayerType::AVG_POOLING:
            return std::make_unique<SerializableAvgPoolingLayer>();
        case LayerType::ACTIVATION:
            return std::make_unique<SerializableActivationLayer>();
        case LayerType::BATCH_NORM:
            return std::make_unique<SerializableBatchNormLayer>();
        default:
            throw std::runtime_error("Unknown layer type");
    }
}

} // namespace cnn_cuda
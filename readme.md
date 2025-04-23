# High-Performance CNN Image Processing Pipeline

This project implements a custom CNN inference engine optimized for image processing, built with C++ and CUDA.

## Project Structure

```
cnn-cuda/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── cnn_cuda.h
│   ├── tensor.h
│   ├── layers/
│   │   ├── conv_layer.h
│   │   ├── pooling_layer.h
│   │   ├── activation_layer.h
│   │   └── norm_layer.h
│   └── utils/
│       ├── memory_pool.h
│       └── image_utils.h
├── src/
│   ├── tensor.cpp
│   ├── layers/
│   │   ├── conv_layer.cpp
│   │   ├── pooling_layer.cpp
│   │   ├── activation_layer.cpp
│   │   └── norm_layer.cpp
│   └── utils/
│       ├── memory_pool.cpp
│       └── image_utils.cpp
├── kernels/
│   ├── conv_kernels.cu
│   ├── pooling_kernels.cu
│   ├── activation_kernels.cu
│   ├── norm_kernels.cu
│   └── utils/
│       ├── tensor_ops.cu
│       └── memory_ops.cu
├── examples/
│   ├── image_classification.cpp
│   └── CMakeLists.txt
└── tests/
    ├── test_tensor.cpp
    ├── test_conv.cpp
    ├── test_pooling.cpp
    └── CMakeLists.txt
```
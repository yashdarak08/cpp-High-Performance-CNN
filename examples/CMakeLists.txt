cmake_minimum_required(VERSION 3.10)

# Image classification example
add_executable(image_classification image_classification.cpp)
target_link_libraries(image_classification cnn_cuda)

# Simple test application
add_executable(cnn_test main.cpp)
target_link_libraries(cnn_test cnn_cuda)
cmake_minimum_required(VERSION 3.14)

# Set output directory for tests
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests)

# Include CTest for test discovery
include(CTest)
enable_testing()

# Find Google Test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Add custom preprocessor definitions for tests
add_compile_definitions(TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test_data")

# Add test utilities library
add_library(test_utils STATIC 
    test_utils.cpp
    test_utils.h
)
target_link_libraries(test_utils PUBLIC cnn_cuda)

# Macro for adding tests
macro(add_cnn_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE cnn_cuda test_utils ${GTEST_LIBRARIES} pthread)
    gtest_discover_tests(${name})
    if(CNN_CUDA_ENABLE_CODE_COVERAGE)
        target_compile_options(${name} PRIVATE --coverage)
        target_link_options(${name} PRIVATE --coverage)
    endif()
endmacro()

# Unit tests
add_cnn_test(test_tensor)
add_cnn_test(test_conv)
add_cnn_test(test_pooling)
add_cnn_test(test_activation)
add_cnn_test(test_norm)
add_cnn_test(test_memory_pool)
add_cnn_test(test_image_utils)
add_cnn_test(test_error_handling)

# Integration tests
add_cnn_test(test_cnn_pipeline)
add_cnn_test(test_model_inference)

# Performance tests (not part of ctest)
add_executable(perf_tensor perf_tensor.cpp)
target_link_libraries(perf_tensor PRIVATE cnn_cuda test_utils)

add_executable(perf_conv perf_conv.cpp)
target_link_libraries(perf_conv PRIVATE cnn_cuda test_utils)

add_executable(perf_pooling perf_pooling.cpp)
target_link_libraries(perf_pooling PRIVATE cnn_cuda test_utils)

add_executable(perf_memory perf_memory.cpp)
target_link_libraries(perf_memory PRIVATE cnn_cuda test_utils)
cmake_minimum_required(VERSION 3.10)

# Find GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Add test executables
add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor cnn_cuda ${GTEST_LIBRARIES} pthread)

add_executable(test_conv test_conv.cpp)
target_link_libraries(test_conv cnn_cuda ${GTEST_LIBRARIES} pthread)

add_executable(test_pooling test_pooling.cpp)
target_link_libraries(test_pooling cnn_cuda ${GTEST_LIBRARIES} pthread)

# Add tests
add_test(NAME TensorTest COMMAND test_tensor)
add_test(NAME ConvTest COMMAND test_conv)
add_test(NAME PoolingTest COMMAND test_pooling)
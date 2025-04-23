cmake_minimum_required(VERSION 3.14)

# Project version and metadata
project(cnn_cuda 
    VERSION 0.2.0
    DESCRIPTION "High-performance CNN inference library with CUDA acceleration"
    HOMEPAGE_URL "https://github.com/yourusername/cnn-cuda"
    LANGUAGES CXX CUDA
)

# Configure version info
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/version.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/include/version.h
)

# Options
option(CNN_CUDA_BUILD_EXAMPLES "Build example applications" ON)
option(CNN_CUDA_BUILD_TESTS "Build tests" ON)
option(CNN_CUDA_BUILD_DOCS "Build documentation" OFF)
option(CNN_CUDA_INSTALL "Generate installation target" ON)
option(CNN_CUDA_USE_STATIC_LIBS "Build static libraries" OFF)

# Find CUDA
find_package(CUDA REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION}")

# Find dependencies
find_package(Threads REQUIRED)

# CUDA architectures to build for
set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80" CACHE STRING "CUDA architectures to build for")
message(STATUS "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# C++ settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA settings
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Detect GPU compute capability if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES OFF)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/detect_cuda_arch.cmake)
    detect_cuda_compute_capability()
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "kernels/*.cu")

# Library
if(CNN_CUDA_USE_STATIC_LIBS)
    add_library(cnn_cuda STATIC ${SOURCES} ${CUDA_SOURCES})
else()
    add_library(cnn_cuda SHARED ${SOURCES} ${CUDA_SOURCES})
endif()

# Set properties for the library
set_target_properties(cnn_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
)

# Link dependencies
target_link_libraries(cnn_cuda
    PUBLIC
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_CUDNN_LIBRARIES}
    PRIVATE
        Threads::Threads
)

# Define compile flags
target_compile_options(cnn_cuda
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror -pedantic -O3 -march=native>
)

# Export compile definitions
target_compile_definitions(cnn_cuda
    PUBLIC
        CNN_CUDA_VERSION="${PROJECT_VERSION}"
        CNN_CUDA_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        CNN_CUDA_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        CNN_CUDA_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# Examples
if(CNN_CUDA_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(CNN_CUDA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Documentation
if(CNN_CUDA_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    add_subdirectory(docs)
endif()

# Installation
if(CNN_CUDA_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    # Install headers
    install(DIRECTORY include/
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cnn_cuda
            FILES_MATCHING PATTERN "*.h"
    )

    # Install library
    install(TARGETS cnn_cuda
            EXPORT cnn_cuda-targets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # Create version file
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/cnn_cuda-config-version.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    # Configure package config file
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cnn_cuda-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/cnn_cuda-config.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cnn_cuda
    )

    # Install config files
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/cnn_cuda-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/cnn_cuda-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cnn_cuda
    )

    # Install export targets
    install(EXPORT cnn_cuda-targets
        FILE cnn_cuda-targets.cmake
        NAMESPACE cnn_cuda::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cnn_cuda
    )

    # Install documentation if built
    if(CNN_CUDA_BUILD_DOCS)
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs/html/
                DESTINATION ${CMAKE_INSTALL_DOCDIR}
        )
    endif()
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "CNN CUDA Configuration Summary:")
message(STATUS "  Version:            ${PROJECT_VERSION}")
message(STATUS "  Build type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA version:       ${CUDA_VERSION}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build examples:     ${CNN_CUDA_BUILD_EXAMPLES}")
message(STATUS "  Build tests:        ${CNN_CUDA_BUILD_TESTS}")
message(STATUS "  Build docs:         ${CNN_CUDA_BUILD_DOCS}")
message(STATUS "  Install:            ${CNN_CUDA_INSTALL}")
message(STATUS "  Static libraries:   ${CNN_CUDA_USE_STATIC_LIBS}")
message(STATUS "")
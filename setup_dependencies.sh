#!/bin/bash

# Script to set up dependencies for the CNN CUDA library
# This script should be run from the root of the project

set -e  # Exit immediately if a command exits with a non-zero status

# Print colored output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up dependencies for CNN CUDA Library...${NC}"

# Check if CUDA is installed
echo -e "\nChecking for CUDA installation..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -c2-)
    echo -e "${GREEN}CUDA is installed (version: $CUDA_VERSION)${NC}"
else
    echo -e "${RED}CUDA not found. Please install CUDA Toolkit 10.0 or newer.${NC}"
    echo "Visit: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Check for CMake
echo -e "\nChecking for CMake..."
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n1 | awk '{print $3}')
    echo -e "${GREEN}CMake is installed (version: $CMAKE_VERSION)${NC}"
else
    echo -e "${RED}CMake not found. Please install CMake 3.14 or newer.${NC}"
    echo "Visit: https://cmake.org/download/"
    exit 1
fi

# Check for a C++17 compatible compiler
echo -e "\nChecking for C++17 compatible compiler..."
if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -n1 | awk '{print $3}')
    MAJOR=$(echo $GCC_VERSION | cut -d. -f1)
    if [[ $MAJOR -ge 7 ]]; then
        echo -e "${GREEN}GCC is installed (version: $GCC_VERSION)${NC}"
    else
        echo -e "${YELLOW}GCC version $GCC_VERSION may not fully support C++17. Consider upgrading.${NC}"
    fi
else
    echo -e "${YELLOW}GCC not found. Make sure you have a C++17 compatible compiler.${NC}"
fi

# Install Google Test if needed
echo -e "\nChecking for Google Test..."
if pkg-config --exists gtest; then
    echo -e "${GREEN}Google Test is installed${NC}"
else
    echo -e "${YELLOW}Google Test not found. Installing...${NC}"
    
    # Create a temporary directory
    TEMP_DIR=$(mktemp -d)
    cd $TEMP_DIR
    
    # Clone and build Google Test
    git clone https://github.com/google/googletest.git
    cd googletest
    mkdir build && cd build
    cmake .. -DCMAKE_CXX_STANDARD=17 -DBUILD_GMOCK=ON
    make -j$(nproc)
    sudo make install
    
    # Clean up
    cd $OLDPWD
    rm -rf $TEMP_DIR
    
    echo -e "${GREEN}Google Test installed successfully${NC}"
fi

# Optional: install OpenCV for image-based examples
echo -e "\nChecking for OpenCV (optional)..."
if pkg-config --exists opencv4; then
    OPENCV_VERSION=$(pkg-config --modversion opencv4)
    echo -e "${GREEN}OpenCV is installed (version: $OPENCV_VERSION)${NC}"
elif pkg-config --exists opencv; then
    OPENCV_VERSION=$(pkg-config --modversion opencv)
    echo -e "${GREEN}OpenCV is installed (version: $OPENCV_VERSION)${NC}"
else
    echo -e "${YELLOW}OpenCV not found. Image-based examples will be disabled.${NC}"
    echo "To install OpenCV:"
    echo "  sudo apt-get install libopencv-dev"
fi

# Create a build directory
echo -e "\n${GREEN}Creating build directory...${NC}"
if [ ! -d "build" ]; then
    mkdir build
fi

echo -e "\n${GREEN}Dependencies setup completed!${NC}"
echo -e "You can now build the project using:"
echo -e "  cd build"
echo -e "  cmake .."
echo -e "  make -j\$(nproc)"
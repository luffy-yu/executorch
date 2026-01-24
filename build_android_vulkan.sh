#!/bin/bash
# Build ExecuTorch for Android with Vulkan backend support
# Usage: ./build_android_vulkan.sh

set -e

# Configuration
ANDROID_ABI=${ANDROID_ABI:-arm64-v8a}
ANDROID_PLATFORM=${ANDROID_PLATFORM:-android-26}
BUILD_DIR="build-android-vulkan"

# Use newer CMake from Android SDK if available
if [ -f "$HOME/Android/Sdk/cmake/4.1.2/bin/cmake" ]; then
    export PATH="$HOME/Android/Sdk/cmake/4.1.2/bin:$PATH"
    echo "Using CMake from Android SDK: $(cmake --version | head -1)"
fi

# Source Vulkan SDK for glslc
if [ -f "/home/n10288/vulkansdk/1.4.335.0/setup-env.sh" ]; then
    source /home/n10288/vulkansdk/1.4.335.0/setup-env.sh
    echo "Using Vulkan SDK: $(glslc --version | head -1)"
fi

# Activate conda environment for Python
eval "$(/home/n10288/miniconda3/bin/conda shell.bash hook)"
conda activate etorch
echo "Using Python: $(which python)"

# Check for Android NDK
if [ -z "$ANDROID_NDK" ]; then
    # Try common locations
    if [ -d "$HOME/Android/Sdk/ndk/26.1.10909125" ]; then
        export ANDROID_NDK="$HOME/Android/Sdk/ndk/26.1.10909125"
    elif [ -d "$HOME/Android/Sdk/ndk-bundle" ]; then
        export ANDROID_NDK="$HOME/Android/Sdk/ndk-bundle"
    else
        echo "Error: ANDROID_NDK not set and not found in common locations"
        exit 1
    fi
fi

echo "=============================================="
echo "Building ExecuTorch for Android with Vulkan"
echo "=============================================="
echo "ANDROID_NDK: $ANDROID_NDK"
echo "ANDROID_ABI: $ANDROID_ABI"
echo "ANDROID_PLATFORM: $ANDROID_PLATFORM"
echo "BUILD_DIR: $BUILD_DIR"
echo ""

# Create build directory
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure with CMake
# Note: EXECUTORCH_USE_PYTORCH_HEADERS=OFF to avoid PyTorch dependency for Android
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=$ANDROID_ABI \
    -DANDROID_PLATFORM=$ANDROID_PLATFORM \
    -DCMAKE_INSTALL_PREFIX=cmake-out-android \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -DEXECUTORCH_USE_PYTORCH_HEADERS=OFF \
    -DPYTHON_EXECUTABLE=/home/n10288/miniconda3/envs/etorch/bin/python

# Build
cmake --build . -j$(nproc) --target vulkan_backend
cmake --build . -j$(nproc) --target executor_runner

echo ""
echo "=============================================="
echo "Build complete!"
echo "=============================================="
echo ""
echo "Vulkan backend: $BUILD_DIR/lib/libvulkan_backend.a"
echo "Executor runner: $BUILD_DIR/executor_runner"
echo ""
echo "To test on Android device:"
echo "  adb push $BUILD_DIR/executor_runner /data/local/tmp/"
echo "  adb push fastvlm_vulkan/vision_encoder.pte /data/local/tmp/"
echo "  adb shell '/data/local/tmp/executor_runner --model_path /data/local/tmp/vision_encoder.pte'"

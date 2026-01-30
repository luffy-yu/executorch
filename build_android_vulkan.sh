#!/bin/bash
# Build ExecuTorch for Android with Vulkan backend support
# Including the Vulkan Multimodal Runner for FastVLM
# Usage: ./build_android_vulkan.sh

set -e

# Configuration
ANDROID_ABI=${ANDROID_ABI:-arm64-v8a}
ANDROID_PLATFORM=${ANDROID_PLATFORM:-android-26}
BUILD_DIR="build-android-vulkan"
EXECUTORCH_ROOT="$(cd "$(dirname "$0")" && pwd)"

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
echo "EXECUTORCH_ROOT: $EXECUTORCH_ROOT"
echo ""

# ============================================================================
# Step 1: Build main ExecuTorch with Vulkan backend
# ============================================================================
echo ">>> Step 1: Building main ExecuTorch with Vulkan backend..."
mkdir -p $BUILD_DIR
cd $BUILD_DIR

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
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -DEXECUTORCH_USE_PYTORCH_HEADERS=OFF \
    -DPYTHON_EXECUTABLE=/home/n10288/miniconda3/envs/etorch/bin/python

# Build all required targets
cmake --build . -j$(nproc) --target vulkan_backend
cmake --build . -j$(nproc) --target executor_runner
# Build extension libraries needed by the multimodal runner
cmake --build . -j$(nproc) --target extension_module || true
cmake --build . -j$(nproc) --target extension_tensor || true
cmake --build . -j$(nproc) --target extension_data_loader || true
cmake --build . -j$(nproc) --target extension_flat_tensor || true
cmake --build . -j$(nproc) --target extension_llm_runner || true
cmake --build . -j$(nproc) --target extension_named_data_map || true
cmake --build . -j$(nproc) --target gflags_nothreads || true
cmake --build . -j$(nproc) --target portable_ops_lib || true
cmake --build . -j$(nproc) --target tokenizers || true

cd $EXECUTORCH_ROOT

# ============================================================================
# Step 2: Build Vulkan Multimodal Runner
# ============================================================================
echo ""
echo ">>> Step 2: Building Vulkan Multimodal Runner..."

RUNNER_BUILD_DIR="${BUILD_DIR}-vulkan-runner"
mkdir -p $RUNNER_BUILD_DIR

cmake \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=$ANDROID_ABI \
    -DANDROID_PLATFORM=$ANDROID_PLATFORM \
    -DEXECUTORCH_ROOT=$EXECUTORCH_ROOT \
    -DEXECUTORCH_BUILD_DIR=$EXECUTORCH_ROOT/$BUILD_DIR \
    -S examples/qualcomm/oss_scripts/llama/vulkan_runner \
    -B $RUNNER_BUILD_DIR

cmake --build $RUNNER_BUILD_DIR -j$(nproc)

echo ""
echo "=============================================="
echo "Build complete!"
echo "=============================================="
echo ""
echo "Main build:           $BUILD_DIR/"
echo "Executor runner:      $BUILD_DIR/executor_runner"
echo "Vulkan multimodal:    $RUNNER_BUILD_DIR/vulkan_multimodal_runner"
echo ""
echo "To test on Android device:"
echo "  adb push $RUNNER_BUILD_DIR/vulkan_multimodal_runner /data/local/tmp/"
echo "  adb push fastvlm_vulkan/*.pte /data/local/tmp/fastvlm/"
echo "  adb push fastvlm_0_5b_hybrid/tokenizer.json /data/local/tmp/fastvlm/"
echo "  adb push 000000039769.jpg /data/local/tmp/fastvlm/"
echo "  adb shell '/data/local/tmp/vulkan_multimodal_runner \\"
echo "    --encoder_path /data/local/tmp/fastvlm/vision_encoder.pte \\"
echo "    --embedding_path /data/local/tmp/fastvlm/text_embedding.pte \\"
echo "    --decoder_path /data/local/tmp/fastvlm/decoder.pte \\"
echo "    --tokenizer_path /data/local/tmp/fastvlm/tokenizer.json \\"
echo "    --image_path /data/local/tmp/fastvlm/000000039769.jpg \\"
echo "    --prompt \"can you describe this image\"'"

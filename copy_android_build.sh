#!/bin/bash

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <output_folder>"
    exit 1
fi

OUTPUT_DIR="$1"
BUILD_DIR="${EXECUTORCH_ROOT:-$(pwd)}/build-android"

mkdir -p "$OUTPUT_DIR"

# Copy QNN SDK .so files
if [ -z "$QNN_SDK_ROOT" ]; then
    echo "Warning: QNN_SDK_ROOT is not set, skipping QNN SDK .so files"
else
    cp "$QNN_SDK_ROOT"/lib/aarch64-android/libQnnHtp.so "$OUTPUT_DIR"
    cp "$QNN_SDK_ROOT"/lib/aarch64-android/libQnnSystem.so "$OUTPUT_DIR"
    cp "$QNN_SDK_ROOT"/lib/aarch64-android/libQnnHtpV69Stub.so "$OUTPUT_DIR"
    cp "$QNN_SDK_ROOT"/lib/aarch64-android/libQnnHtpV73Stub.so "$OUTPUT_DIR"
    cp "$QNN_SDK_ROOT"/lib/aarch64-android/libQnnHtpV75Stub.so "$OUTPUT_DIR"
    cp "$QNN_SDK_ROOT"/lib/aarch64-android/libQnnHtpV79Stub.so "$OUTPUT_DIR"
    cp "$QNN_SDK_ROOT"/lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so "$OUTPUT_DIR"
    cp "$QNN_SDK_ROOT"/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so "$OUTPUT_DIR"
    cp "$QNN_SDK_ROOT"/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so "$OUTPUT_DIR"
    cp "$QNN_SDK_ROOT"/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so "$OUTPUT_DIR"
    echo "Copied QNN SDK .so files"
fi

# Copy executorch backend .so
cp "$BUILD_DIR"/backends/qualcomm/libqnn_executorch_backend.so "$OUTPUT_DIR"
echo "Copied libqnn_executorch_backend.so"

# Copy qnn_multimodal_runner
cp "$BUILD_DIR"/examples/qualcomm/oss_scripts/llama/qnn_multimodal_runner "$OUTPUT_DIR"
echo "Copied qnn_multimodal_runner"

echo "All files copied to $OUTPUT_DIR"

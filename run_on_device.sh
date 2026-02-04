#!/bin/bash

set -e

usage() {
    echo "Usage: $0 <local_folder> <device_dir> <decoder_model_version> <image_path> <prompt>"
    echo ""
    echo "  local_folder           Local folder containing build artifacts (.so, runner, .pte, tokenizer, image)"
    echo "  device_dir             Target directory on the Android device (e.g. /data/local/tmp/fastvlm)"
    echo "  decoder_model_version  Model version (e.g. smolvlm, internvl3, fastvlm)"
    echo "  image_path             Path to the image file (relative to local_folder)"
    echo "  prompt                 Prompt string for the model"
    exit 1
}

if [ $# -lt 5 ]; then
    usage
fi

LOCAL_FOLDER="$1"
DEVICE_DIR="$2"
DECODER_MODEL_VERSION="$3"
IMAGE_PATH="$4"
PROMPT="$5"

# Create device directory
adb shell "mkdir -p ${DEVICE_DIR}"

# Push all files from local folder to device
echo "Pushing files from ${LOCAL_FOLDER} to ${DEVICE_DIR}..."
adb push "${LOCAL_FOLDER}/." "${DEVICE_DIR}"
echo "All files pushed"

# Make runner executable
adb shell "chmod +x ${DEVICE_DIR}/qnn_multimodal_runner"

# Run inference
echo "Running inference..."
adb shell "cd ${DEVICE_DIR} && ./qnn_multimodal_runner \
    --decoder_model_version ${DECODER_MODEL_VERSION} \
    --tokenizer_path tokenizer.json \
    --decoder_path hybrid_llama_qnn.pte \
    --encoder_path vision_encoder_qnn.pte \
    --embedding_path text_embedding_qnn.pte \
    --image_path ${IMAGE_PATH} \
    --seq_len 1024 \
    --output_path outputs.txt \
    --performance_output_path inference_speed.txt \
    --shared_buffer \
    --prompt \"${PROMPT}\" \
    --eval_mode 1 \
    --temperature 0.8 \
    --system_prompt '' \
    && cat outputs.txt"

/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/encoder.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>

using executorch::aten::Tensor;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::Error;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;

namespace example {

namespace {

/**
 * @brief Check if a string ends with a given suffix (case-insensitive)
 */
bool ends_with_ci(const std::string& str, const std::string& suffix) {
  if (suffix.size() > str.size()) {
    return false;
  }
  auto str_lower = str;
  auto suffix_lower = suffix;
  std::transform(
      str_lower.begin(), str_lower.end(), str_lower.begin(), ::tolower);
  std::transform(
      suffix_lower.begin(), suffix_lower.end(), suffix_lower.begin(), ::tolower);
  return str_lower.compare(
             str_lower.size() - suffix_lower.size(),
             suffix_lower.size(),
             suffix_lower) == 0;
}

/**
 * @brief Check if a file is an image file based on its extension
 */
bool is_image_file(const std::string& file_path) {
  return ends_with_ci(file_path, ".jpg") || ends_with_ci(file_path, ".jpeg") ||
      ends_with_ci(file_path, ".png") || ends_with_ci(file_path, ".bmp");
}

/**
 * @brief Load an image file and convert it to a float buffer in CHW format
 *
 * This function mimics the behavior of CLIPImageProcessor used in FastVLM:
 * 1. Loads the image using stb_image
 * 2. Resizes so the shortest edge matches the target size (maintaining aspect ratio)
 * 3. Center-crops to the target dimensions
 * 4. Converts from HWC to CHW format
 * 5. Normalizes pixel values from [0, 255] to [0.0, 1.0]
 *
 * @param image_path Path to the image file
 * @param target_height Target height for the output tensor
 * @param target_width Target width for the output tensor
 * @param target_channels Expected number of channels (usually 3 for RGB)
 * @return Vector of float values in CHW format, or empty vector on failure
 */
std::vector<float> load_image_to_chw_buffer(
    const std::string& image_path,
    int target_height,
    int target_width,
    int target_channels) {
  int width, height, channels;
  unsigned char* data =
      stbi_load(image_path.c_str(), &width, &height, &channels, 0);

  if (!data) {
    ET_LOG(Error, "Failed to load image: %s", image_path.c_str());
    return {};
  }

  ET_LOG(
      Info,
      "Loaded image: %s, original size: %dx%d, channels: %d",
      image_path.c_str(),
      width,
      height,
      channels);

  // If channels don't match, we need to handle the conversion
  if (channels != target_channels) {
    ET_LOG(
        Info,
        "Converting from %d channels to %d channels",
        channels,
        target_channels);
    // Reload with forced channel count
    stbi_image_free(data);
    data = stbi_load(
        image_path.c_str(), &width, &height, &channels, target_channels);
    if (!data) {
      ET_LOG(
          Error,
          "Failed to load image with %d channels: %s",
          target_channels,
          image_path.c_str());
      return {};
    }
    channels = target_channels;
  }

  // Step 1: Resize so that the shortest edge equals the target size
  // This mimics CLIPImageProcessor's size={"shortest_edge": target_size} behavior
  int shortest_edge = std::min(target_height, target_width);
  float scale;
  int new_width, new_height;

  if (width < height) {
    // Width is shorter, scale based on width
    scale = static_cast<float>(shortest_edge) / width;
    new_width = shortest_edge;
    new_height = static_cast<int>(std::round(height * scale));
  } else {
    // Height is shorter (or equal), scale based on height
    scale = static_cast<float>(shortest_edge) / height;
    new_height = shortest_edge;
    new_width = static_cast<int>(std::round(width * scale));
  }

  ET_LOG(
      Info,
      "Resizing with shortest edge scaling: %dx%d -> %dx%d (scale=%.4f)",
      width,
      height,
      new_width,
      new_height,
      scale);

  // Resize to intermediate size (maintaining aspect ratio)
  std::vector<uint8_t> resized_data(new_width * new_height * channels);
  int resize_result = stbir_resize_uint8(
      data,
      width,
      height,
      0,
      resized_data.data(),
      new_width,
      new_height,
      0,
      channels);

  stbi_image_free(data);

  if (!resize_result) {
    ET_LOG(Error, "Failed to resize image");
    return {};
  }

  // Step 2: Center crop to target dimensions
  // This mimics CLIPImageProcessor's crop_size behavior
  int crop_x = (new_width - target_width) / 2;
  int crop_y = (new_height - target_height) / 2;

  // Ensure crop coordinates are non-negative
  crop_x = std::max(0, crop_x);
  crop_y = std::max(0, crop_y);

  ET_LOG(
      Info,
      "Center cropping from (%d, %d) to get %dx%d",
      crop_x,
      crop_y,
      target_width,
      target_height);

  // Step 3: Convert from HWC to CHW and normalize to [0, 1]
  // This mimics rescale_factor=1.0/255.0 with no mean/std normalization
  std::vector<float> chw_data(channels * target_height * target_width);

  for (int h = 0; h < target_height; ++h) {
    for (int w = 0; w < target_width; ++w) {
      int src_h = crop_y + h;
      int src_w = crop_x + w;

      // Handle edge cases where crop might exceed bounds
      if (src_h >= new_height)
        src_h = new_height - 1;
      if (src_w >= new_width)
        src_w = new_width - 1;

      for (int c = 0; c < channels; ++c) {
        uint8_t pixel_value =
            resized_data[src_h * new_width * channels + src_w * channels + c];
        chw_data[c * target_height * target_width + h * target_width + w] =
            static_cast<float>(pixel_value) / 255.0f;
      }
    }
  }

  ET_LOG(
      Info,
      "Converted image to CHW format (float32), total elements: %zu",
      chw_data.size());

  return chw_data;
}

} // namespace

EncoderRunner::EncoderRunner(const std::string& model_path)
    : image_seq_len_(0) {
  module_ = std::make_unique<Module>(
      model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);
  ET_LOG(Info, "Creating encoder module: model_path=%s", model_path.c_str());
}

bool EncoderRunner::is_method_loaded() const {
  return module_->is_method_loaded(kEncoderForwardName);
}

Error EncoderRunner::load() {
  if (is_method_loaded()) {
    return Error::Ok;
  }

  auto load_result = module_->load_method(kEncoderForwardName);
  if (load_result != Error::Ok) {
    ET_LOG(Error, "Failed to load encoder method");
    return load_result;
  }

  // Get image sequence length from output metadata
  Result<MethodMeta> method_meta = module_->method_meta(kEncoderForwardName);
  if (!method_meta.ok()) {
    ET_LOG(Error, "Failed to get encoder method metadata");
    return method_meta.error();
  }

  // vision embedding output shape: [1, seq_len, dim]
  image_seq_len_ = method_meta->output_tensor_meta(0)->sizes()[1];
  ET_LOG(Info, "Encoder loaded successfully, image_seq_len=%d", image_seq_len_);

  return Error::Ok;
}

int32_t EncoderRunner::get_image_seq_len() const {
  return image_seq_len_;
}

Result<Tensor> EncoderRunner::encode(TensorPtr& image_tensor) {
  ET_CHECK_MSG(is_method_loaded(), "Encoder method not loaded");

  auto tensor_ptr = image_tensor.get();
  ET_LOG(Info, "Encoding image tensor with numel: %zu", tensor_ptr->numel());

  std::vector<executorch::runtime::EValue> encoder_inputs;
  encoder_inputs.emplace_back(*tensor_ptr);

  auto encoder_result = module_->forward(encoder_inputs);
  ET_CHECK_MSG(encoder_result.ok(), "Encoder execution failed");

  auto encoder_output = encoder_result.get();
  auto image_hidden_states = encoder_output[0].toTensor();
  ET_LOG(Info, "Encoder execution completed, got image hidden states");

  return image_hidden_states;
}

Result<Tensor> EncoderRunner::encode_from_file(
    const std::string& image_file_path) {
  ET_CHECK_MSG(is_method_loaded(), "Encoder method not loaded");

  // Get input tensor metadata
  Result<MethodMeta> method_meta = module_->method_meta(kEncoderForwardName);
  auto sizes_span = method_meta->input_tensor_meta(0)->sizes();

  // Calculate total number of elements
  int64_t num_elem = 1;
  for (const auto& size : sizes_span) {
    num_elem *= size;
  }

  // Extract dimensions from input tensor shape [batch, channels, height, width]
  ET_CHECK_MSG(
      sizes_span.size() == 4,
      "Expected 4D input tensor [B, C, H, W], got %zu dimensions",
      sizes_span.size());
  int target_channels = static_cast<int>(sizes_span[1]);
  int target_height = static_cast<int>(sizes_span[2]);
  int target_width = static_cast<int>(sizes_span[3]);

  ET_LOG(
      Info,
      "Encoder input shape: [%ld, %d, %d, %d], num_elements=%ld",
      sizes_span[0],
      target_channels,
      target_height,
      target_width,
      num_elem);

  std::vector<float> buffer;

  // Check if the input is an image file (jpg, png, bmp) or a raw binary file
  if (is_image_file(image_file_path)) {
    ET_LOG(
        Info,
        "Detected image file: %s, loading and preprocessing...",
        image_file_path.c_str());

    buffer = load_image_to_chw_buffer(
        image_file_path, target_height, target_width, target_channels);

    ET_CHECK_MSG(
        !buffer.empty(),
        "Failed to load image file: %s",
        image_file_path.c_str());

    ET_CHECK_MSG(
        static_cast<int64_t>(buffer.size()) == num_elem,
        "Image buffer size mismatch: expected %ld elements but got %zu elements",
        num_elem,
        buffer.size());
  } else {
    // Read raw binary data from file (original behavior)
    ET_LOG(
        Info,
        "Reading raw binary from file: %s, num_elements=%ld",
        image_file_path.c_str(),
        num_elem);
    std::ifstream file(image_file_path, std::ios::binary | std::ios::ate);
    ET_CHECK_MSG(
        file.is_open(),
        "Failed to open image file: %s",
        image_file_path.c_str());

    // To prevent users from passing images that have not been
    // resized to match the encoder input size.
    std::streamsize file_size = file.tellg();
    std::streamsize expected_size = num_elem * sizeof(float);
    ET_CHECK_MSG(
        file_size == expected_size,
        "Image file size mismatch: expected %ld bytes but got %ld bytes (file: %s). "
        "Hint: You can now pass JPG/PNG/BMP image files directly using --image_path.",
        expected_size,
        file_size,
        image_file_path.c_str());

    file.seekg(0, std::ios::beg);
    buffer.resize(num_elem);
    file.read(reinterpret_cast<char*>(buffer.data()), expected_size);
    file.close();
  }

  // DEBUG: Print input statistics
  float min_val = buffer[0], max_val = buffer[0];
  double sum = 0.0, sum_sq = 0.0;
  for (int64_t i = 0; i < num_elem; ++i) {
    float val = buffer[i];
    min_val = std::min(min_val, val);
    max_val = std::max(max_val, val);
    sum += val;
    sum_sq += val * val;
  }
  double mean = sum / num_elem;
  double variance = (sum_sq / num_elem) - (mean * mean);
  double std_dev = std::sqrt(variance > 0 ? variance : 0);
  ET_LOG(
      Info,
      "[DEBUG] Runtime vision encoder input: num_elem=%ld, range=[%.4f, %.4f], mean=%.4f, std=%.4f",
      num_elem,
      min_val,
      max_val,
      mean,
      std_dev);

  // DEBUG: Save runtime input to file for comparison
  std::ofstream debug_input("debug_runtime_vision_input.raw", std::ios::binary);
  if (debug_input.is_open()) {
    debug_input.write(
        reinterpret_cast<char*>(buffer.data()), num_elem * sizeof(float));
    debug_input.close();
    ET_LOG(
        Info, "[DEBUG] Saved runtime vision input to debug_runtime_vision_input.raw");
  }

  // Create tensor from buffer
  TensorPtr tensor = executorch::extension::from_blob(
      buffer.data(),
      std::vector<int32_t>(sizes_span.begin(), sizes_span.end()),
      executorch::aten::ScalarType::Float);

  // Encode the tensor
  auto result = encode(tensor);

  // DEBUG: Save and print output statistics
  if (result.ok()) {
    auto output_tensor = result.get();
    const float* output_data = output_tensor.const_data_ptr<float>();
    int64_t output_numel = output_tensor.numel();

    float out_min = output_data[0], out_max = output_data[0];
    double out_sum = 0.0, out_sum_sq = 0.0;
    for (int64_t i = 0; i < output_numel; ++i) {
      float val = output_data[i];
      out_min = std::min(out_min, val);
      out_max = std::max(out_max, val);
      out_sum += val;
      out_sum_sq += val * val;
    }
    double out_mean = out_sum / output_numel;
    double out_variance = (out_sum_sq / output_numel) - (out_mean * out_mean);
    double out_std = std::sqrt(out_variance > 0 ? out_variance : 0);
    ET_LOG(
        Info,
        "[DEBUG] Runtime vision encoder output: numel=%ld, range=[%.4f, %.4f], mean=%.4f, std=%.4f",
        output_numel,
        out_min,
        out_max,
        out_mean,
        out_std);

    // Save output to file
    std::ofstream debug_output("debug_runtime_vision_output.raw", std::ios::binary);
    if (debug_output.is_open()) {
      debug_output.write(
          reinterpret_cast<const char*>(output_data),
          output_numel * sizeof(float));
      debug_output.close();
      ET_LOG(Info, "[DEBUG] Saved runtime vision output to debug_runtime_vision_output.raw");
    }
  }

  return result;
}

} // namespace example

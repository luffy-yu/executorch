/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

#include <executorch/extension/android/jni/jni_helper.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

#if defined(EXECUTORCH_BUILD_QNN)
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#endif

#if defined(EXECUTORCH_BUILD_QNN_MULTIMODAL)
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/encoder.h>
#endif

#if defined(EXECUTORCH_BUILD_MEDIATEK)
#include <executorch/examples/mediatek/executor_runner/mtk_llama_runner.h>
#endif

namespace llm = ::executorch::extension::llm;
using ::executorch::runtime::Error;

namespace {
bool utf8_check_validity(const char* str, size_t length) {
  for (size_t i = 0; i < length; ++i) {
    uint8_t byte = static_cast<uint8_t>(str[i]);
    if (byte >= 0x80) { // Non-ASCII byte
      if (i + 1 >= length) { // Incomplete sequence
        return false;
      }
      uint8_t next_byte = static_cast<uint8_t>(str[i + 1]);
      if ((byte & 0xE0) == 0xC0 &&
          (next_byte & 0xC0) == 0x80) { // 2-byte sequence
        i += 1;
      } else if (
          (byte & 0xF0) == 0xE0 && (next_byte & 0xC0) == 0x80 &&
          (i + 2 < length) &&
          (static_cast<uint8_t>(str[i + 2]) & 0xC0) ==
              0x80) { // 3-byte sequence
        i += 2;
      } else if (
          (byte & 0xF8) == 0xF0 && (next_byte & 0xC0) == 0x80 &&
          (i + 2 < length) &&
          (static_cast<uint8_t>(str[i + 2]) & 0xC0) == 0x80 &&
          (i + 3 < length) &&
          (static_cast<uint8_t>(str[i + 3]) & 0xC0) ==
              0x80) { // 4-byte sequence
        i += 3;
      } else {
        return false; // Invalid sequence
      }
    }
  }
  return true; // All bytes were valid
}

std::string token_buffer;
} // namespace

namespace executorch_jni {

class ExecuTorchLlmCallbackJni
    : public facebook::jni::JavaClass<ExecuTorchLlmCallbackJni> {
 public:
  constexpr static const char* kJavaDescriptor =
      "Lorg/pytorch/executorch/extension/llm/LlmCallback;";

  void onResult(std::string result) const {
    static auto cls = ExecuTorchLlmCallbackJni::javaClassStatic();
    static const auto method =
        cls->getMethod<void(facebook::jni::local_ref<jstring>)>("onResult");

    token_buffer += result;
    if (!utf8_check_validity(token_buffer.c_str(), token_buffer.size())) {
      ET_LOG(
          Info, "Current token buffer is not valid UTF-8. Waiting for more.");
      return;
    }
    result = token_buffer;
    token_buffer = "";
    facebook::jni::local_ref<jstring> s = facebook::jni::make_jstring(result);
    method(self(), s);
  }

  void onStats(const llm::Stats& result) const {
    static auto cls = ExecuTorchLlmCallbackJni::javaClassStatic();
    static const auto on_stats_method =
        cls->getMethod<void(facebook::jni::local_ref<jstring>)>("onStats");
    on_stats_method(
        self(),
        facebook::jni::make_jstring(
            executorch::extension::llm::stats_to_json_string(result)));
  }
};

class ExecuTorchLlmJni : public facebook::jni::HybridClass<ExecuTorchLlmJni> {
 private:
  friend HybridBase;
  float temperature_ = 0.0f;
  int model_type_category_;
  std::unique_ptr<llm::IRunner> runner_;
  std::unique_ptr<executorch::extension::llm::MultimodalRunner>
      multi_modal_runner_;
  std::vector<llm::MultimodalInput> prefill_inputs_;
#if defined(EXECUTORCH_BUILD_QNN_MULTIMODAL)
  std::unique_ptr<example::MultimodalRunner<uint16_t>> qnn_multimodal_runner_;
  std::unique_ptr<example::EncoderRunner> encoder_runner_;
  std::unique_ptr<executorch::aten::Tensor> image_hidden_states_;
#endif

 public:
  constexpr static auto kJavaDescriptor =
      "Lorg/pytorch/executorch/extension/llm/LlmModule;";

  constexpr static int MODEL_TYPE_CATEGORY_LLM = 1;
  constexpr static int MODEL_TYPE_CATEGORY_MULTIMODAL = 2;
  constexpr static int MODEL_TYPE_MEDIATEK_LLAMA = 3;
  constexpr static int MODEL_TYPE_QNN_LLAMA = 4;
  constexpr static int MODEL_TYPE_QNN_MULTIMODAL = 5;

  static facebook::jni::local_ref<jhybriddata> initHybrid(
      facebook::jni::alias_ref<jclass>,
      jint model_type_category,
      facebook::jni::alias_ref<jstring> model_path,
      facebook::jni::alias_ref<jstring> tokenizer_path,
      jfloat temperature,
      facebook::jni::alias_ref<facebook::jni::JList<jstring>::javaobject>
          data_files) {
    return makeCxxInstance(
        model_type_category,
        model_path,
        tokenizer_path,
        temperature,
        data_files);
  }

  ExecuTorchLlmJni(
      jint model_type_category,
      facebook::jni::alias_ref<jstring> model_path,
      facebook::jni::alias_ref<jstring> tokenizer_path,
      jfloat temperature,
      facebook::jni::alias_ref<jobject> data_files = nullptr) {
    temperature_ = temperature;
#if defined(ET_USE_THREADPOOL)
    // Reserve 1 thread for the main thread.
    int32_t num_performant_cores =
        ::executorch::extension::cpuinfo::get_num_performant_cores() - 1;
    if (num_performant_cores > 0) {
      ET_LOG(Info, "Resetting threadpool to %d threads", num_performant_cores);
      ::executorch::extension::threadpool::get_threadpool()
          ->_unsafe_reset_threadpool(num_performant_cores);
    }
#endif

    model_type_category_ = model_type_category;
    std::vector<std::string> data_files_vector;

    // Convert Java List<String> to C++ std::vector<string> for all model types
    if (data_files != nullptr) {
      auto list_class = facebook::jni::findClassStatic("java/util/List");
      auto size_method = list_class->getMethod<jint()>("size");
      auto get_method =
          list_class->getMethod<facebook::jni::local_ref<jobject>(jint)>(
              "get");

      jint size = size_method(data_files);
      ET_LOG(Info, "JNI: Parsing %d dataFiles entries", size);
      for (jint i = 0; i < size; ++i) {
        auto str_obj = get_method(data_files, i);
        auto jstr = facebook::jni::static_ref_cast<jstring>(str_obj);
        std::string entry = jstr->toStdString();
        ET_LOG(Info, "JNI: dataFiles[%d] = %s", i, entry.c_str());
        data_files_vector.push_back(entry);
      }
    }

    if (model_type_category == MODEL_TYPE_CATEGORY_MULTIMODAL) {
      multi_modal_runner_ = llm::create_multimodal_runner(
          model_path->toStdString().c_str(),
          llm::load_tokenizer(tokenizer_path->toStdString()));
    } else if (model_type_category == MODEL_TYPE_CATEGORY_LLM) {
      runner_ = executorch::extension::llm::create_text_llm_runner(
          model_path->toStdString(),
          llm::load_tokenizer(tokenizer_path->toStdString()),
          data_files_vector);
#if defined(EXECUTORCH_BUILD_QNN)
    } else if (model_type_category == MODEL_TYPE_QNN_LLAMA) {
      std::unique_ptr<executorch::extension::Module> module = std::make_unique<
          executorch::extension::Module>(
          model_path->toStdString().c_str(),
          data_files_vector,
          executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
      std::string decoder_model = "llama3"; // use llama3 for now
      runner_ = std::make_unique<example::Runner<uint16_t>>( // QNN runner
          std::move(module),
          decoder_model.c_str(),
          model_path->toStdString().c_str(),
          tokenizer_path->toStdString().c_str(),
          "",
          "");
      model_type_category_ = MODEL_TYPE_CATEGORY_LLM;
#endif
#if defined(EXECUTORCH_BUILD_QNN_MULTIMODAL)
    } else if (model_type_category == MODEL_TYPE_QNN_MULTIMODAL) {
      // Parse data_files for encoder_path, embedding_path, and config
      // Expected format in data_files:
      // [0]: encoder_path
      // [1]: embedding_path
      // [2]: config string (decoder_model_version:fastvlm_0_5b;eval_mode:1)
      std::string encoder_path;
      std::string embedding_path;
      std::string decoder_model_version = "fastvlm_0_5b";
      int eval_mode = 1;

      if (data_files_vector.size() >= 1) {
        encoder_path = data_files_vector[0];
      }
      if (data_files_vector.size() >= 2) {
        embedding_path = data_files_vector[1];
      }
      if (data_files_vector.size() >= 3) {
        // Parse config string
        std::string config = data_files_vector[2];
        // Parse decoder_model_version
        size_t pos = config.find("decoder_model_version:");
        if (pos != std::string::npos) {
          size_t end = config.find(";", pos);
          decoder_model_version = config.substr(pos + 22,
              end == std::string::npos ? std::string::npos : end - pos - 22);
        }
        // Parse eval_mode
        pos = config.find("eval_mode:");
        if (pos != std::string::npos) {
          eval_mode = std::stoi(config.substr(pos + 10, 1));
        }
      }

      ET_LOG(Info, "QNN Multimodal: decoder=%s, encoder=%s, embedding=%s, model=%s, eval_mode=%d",
             model_path->toStdString().c_str(),
             encoder_path.c_str(),
             embedding_path.c_str(),
             decoder_model_version.c_str(),
             eval_mode);

      // Validate paths are not empty
      if (encoder_path.empty()) {
        ET_LOG(Error, "Encoder path is empty - check dataFiles[0]");
        throw std::runtime_error("Encoder path is empty for QNN multimodal");
      }
      if (embedding_path.empty()) {
        ET_LOG(Error, "Embedding path is empty - check dataFiles[1]");
        throw std::runtime_error("Embedding path is empty for QNN multimodal");
      }

      // Create encoder runner
      ET_LOG(Info, "Creating encoder runner with path: %s", encoder_path.c_str());
      encoder_runner_ = std::make_unique<example::EncoderRunner>(encoder_path);

      // Create embedding module
      std::unique_ptr<executorch::extension::Module> embedding_module =
          std::make_unique<executorch::extension::Module>(
              embedding_path.c_str(),
              executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);

      // Create decoder module
      std::unique_ptr<executorch::extension::Module> decoder_module =
          std::make_unique<executorch::extension::Module>(
              model_path->toStdString().c_str(),
              executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);

      // Create QNN multimodal runner (image_hidden_states will be set later)
      ET_LOG(Info, "Creating QNN multimodal runner...");
      qnn_multimodal_runner_ = std::make_unique<example::MultimodalRunner<uint16_t>>(
          std::move(decoder_module),
          std::move(embedding_module),
          decoder_model_version.c_str(),
          model_path->toStdString().c_str(),
          tokenizer_path->toStdString().c_str(),
          "", // performance_output_path
          "", // dump_logits_path
          temperature,
          eval_mode,
          false, // shared_buffer
          0, // ngram
          0, // window
          0, // gcap
          nullptr); // image_hidden_states set during generate
      model_type_category_ = MODEL_TYPE_QNN_MULTIMODAL;
      ET_LOG(Info, "QNN multimodal runner created successfully");
#endif
#if defined(EXECUTORCH_BUILD_MEDIATEK)
    } else if (model_type_category == MODEL_TYPE_MEDIATEK_LLAMA) {
      runner_ = std::make_unique<MTKLlamaRunner>(
          model_path->toStdString().c_str(),
          tokenizer_path->toStdString().c_str());
      // Interpret the model type as LLM
      model_type_category_ = MODEL_TYPE_CATEGORY_LLM;
#endif
    }
  }

  jint generate(
      facebook::jni::alias_ref<jstring> prompt,
      jint seq_len,
      facebook::jni::alias_ref<ExecuTorchLlmCallbackJni> callback,
      jboolean echo) {
    if (model_type_category_ == MODEL_TYPE_CATEGORY_MULTIMODAL) {
      std::vector<llm::MultimodalInput> inputs = prefill_inputs_;
      prefill_inputs_.clear();
      if (!prompt->toStdString().empty()) {
        inputs.emplace_back(llm::MultimodalInput{prompt->toStdString()});
      }
      executorch::extension::llm::GenerationConfig config{
          .echo = static_cast<bool>(echo),
          .seq_len = seq_len,
          .temperature = temperature_,
      };
      multi_modal_runner_->generate(
          std::move(inputs),
          config,
          [callback](const std::string& result) { callback->onResult(result); },
          [callback](const llm::Stats& result) { callback->onStats(result); });
    } else if (model_type_category_ == MODEL_TYPE_CATEGORY_LLM) {
      executorch::extension::llm::GenerationConfig config{
          .echo = static_cast<bool>(echo),
          .seq_len = seq_len,
          .temperature = temperature_,
      };
      runner_->generate(
          prompt->toStdString(),
          config,
          [callback](std::string result) { callback->onResult(result); },
          [callback](const llm::Stats& result) { callback->onStats(result); });
#if defined(EXECUTORCH_BUILD_QNN_MULTIMODAL)
    } else if (model_type_category_ == MODEL_TYPE_QNN_MULTIMODAL) {
      // Check if image has been encoded
      if (image_hidden_states_ == nullptr) {
        ET_LOG(Error, "No image has been encoded. Call encodeImageFromFile first.");
        return static_cast<jint>(Error::InvalidState);
      }

      // Pass the encoded image hidden states to the runner
      qnn_multimodal_runner_->set_image_hidden_states(std::move(image_hidden_states_));

      executorch::extension::llm::GenerationConfig config{
          .echo = static_cast<bool>(echo),
          .seq_len = seq_len,
          .temperature = temperature_,
      };

      qnn_multimodal_runner_->generate(
          prompt->toStdString(),
          config,
          [callback](const std::string& result) { callback->onResult(result); },
          [callback](const llm::Stats& result) { callback->onStats(result); });
#endif
    }
    return 0;
  }

  // Returns status_code
  // Contract is valid within an AAR (JNI + corresponding Java code)
  jint append_text_input(facebook::jni::alias_ref<jstring> prompt) {
    prefill_inputs_.emplace_back(llm::MultimodalInput{prompt->toStdString()});
    return 0;
  }

  // Returns status_code
  jint append_images_input(
      facebook::jni::alias_ref<jintArray> image,
      jint width,
      jint height,
      jint channels) {
    std::vector<llm::Image> images;
    if (image == nullptr) {
      return static_cast<jint>(Error::EndOfMethod);
    }
    auto image_size = image->size();
    if (image_size != 0) {
      std::vector<jint> image_data_jint(image_size);
      std::vector<uint8_t> image_data(image_size);
      image->getRegion(0, image_size, image_data_jint.data());
      for (int i = 0; i < image_size; i++) {
        image_data[i] = image_data_jint[i];
      }
      llm::Image image_runner{std::move(image_data), width, height, channels};
      prefill_inputs_.emplace_back(
          llm::MultimodalInput{std::move(image_runner)});
    }

    return 0;
  }

  // Returns status_code
  jint append_normalized_images_input(
      facebook::jni::alias_ref<jfloatArray> image,
      jint width,
      jint height,
      jint channels) {
    std::vector<llm::Image> images;
    if (image == nullptr) {
      return static_cast<jint>(Error::EndOfMethod);
    }
    auto image_size = image->size();
    if (image_size != 0) {
      std::vector<jfloat> image_data_jfloat(image_size);
      std::vector<float> image_data(image_size);
      image->getRegion(0, image_size, image_data_jfloat.data());
      for (int i = 0; i < image_size; i++) {
        image_data[i] = image_data_jfloat[i];
      }
      llm::Image image_runner{std::move(image_data), width, height, channels};
      prefill_inputs_.emplace_back(
          llm::MultimodalInput{std::move(image_runner)});
    }

    return 0;
  }

#if defined(EXECUTORCH_BUILD_QNN_MULTIMODAL)
  // Encode image from raw file for QNN multimodal
  // This loads a pre-processed .raw image file and runs the encoder
  jint encode_image_from_file(facebook::jni::alias_ref<jstring> image_path) {
    if (model_type_category_ != MODEL_TYPE_QNN_MULTIMODAL || encoder_runner_ == nullptr) {
      ET_LOG(Error, "encode_image_from_file only supported for QNN multimodal");
      return static_cast<jint>(Error::InvalidArgument);
    }

    std::string path = image_path->toStdString();
    ET_LOG(Info, "Encoding image from file: %s", path.c_str());

    auto result = encoder_runner_->encode_from_file(path);
    if (!result.ok()) {
      ET_LOG(Error, "Failed to encode image from file");
      return static_cast<jint>(result.error());
    }

    // Store the image hidden states for use in generation
    image_hidden_states_ = std::make_unique<executorch::aten::Tensor>(result.get());
    ET_LOG(Info, "Image encoded successfully, hidden states stored");
    return 0;
  }

  // Encode image from normalized float array for QNN multimodal
  // Image should be preprocessed: resized to 1024x1024, normalized with FastVLM mean/std
  // Shape: [batch, channels, height, width] = [1, 3, 1024, 1024]
  jint encode_image(
      facebook::jni::alias_ref<jfloatArray> image,
      jint batch,
      jint channels,
      jint height,
      jint width) {
    if (model_type_category_ != MODEL_TYPE_QNN_MULTIMODAL || encoder_runner_ == nullptr) {
      ET_LOG(Error, "encode_image only supported for QNN multimodal");
      return static_cast<jint>(Error::InvalidArgument);
    }

    if (image == nullptr) {
      ET_LOG(Error, "Image array is null");
      return static_cast<jint>(Error::InvalidArgument);
    }

    auto image_size = image->size();
    int64_t expected_size = batch * channels * height * width;
    if (image_size != expected_size) {
      ET_LOG(Error, "Image size mismatch: expected %ld but got %zu", expected_size, image_size);
      return static_cast<jint>(Error::InvalidArgument);
    }

    // Copy image data
    std::vector<jfloat> image_data_jfloat(image_size);
    image->getRegion(0, image_size, image_data_jfloat.data());

    // Create vector of float for the image data
    std::vector<float> image_buffer(image_data_jfloat.begin(), image_data_jfloat.end());

    ET_LOG(Info, "Encoding image: batch=%d, channels=%d, height=%d, width=%d, numel=%zu",
           batch, channels, height, width, image_size);

    // Create tensor from buffer
    executorch::extension::TensorPtr tensor = executorch::extension::from_blob(
        image_buffer.data(),
        std::vector<int32_t>{batch, channels, height, width},
        executorch::aten::ScalarType::Float);

    // Encode the tensor
    auto result = encoder_runner_->encode(tensor);
    if (!result.ok()) {
      ET_LOG(Error, "Failed to encode image");
      return static_cast<jint>(result.error());
    }

    // Store the image hidden states for use in generation
    image_hidden_states_ = std::make_unique<executorch::aten::Tensor>(result.get());
    ET_LOG(Info, "Image encoded successfully from array, hidden states stored");
    return 0;
  }
#endif

  // Returns status_code
  jint append_audio_input(
      facebook::jni::alias_ref<jbyteArray> data,
      jint batch_size,
      jint n_bins,
      jint n_frames) {
    if (data == nullptr) {
      return static_cast<jint>(Error::EndOfMethod);
    }
    auto data_size = data->size();
    if (data_size != 0) {
      std::vector<jbyte> data_jbyte(data_size);
      std::vector<uint8_t> data_u8(data_size);
      data->getRegion(0, data_size, data_jbyte.data());
      for (int i = 0; i < data_size; i++) {
        data_u8[i] = data_jbyte[i];
      }
      llm::Audio audio{std::move(data_u8), batch_size, n_bins, n_frames};
      prefill_inputs_.emplace_back(llm::MultimodalInput{std::move(audio)});
    }
    return 0;
  }

  // Returns status_code
  jint append_audio_input_float(
      facebook::jni::alias_ref<jfloatArray> data,
      jint batch_size,
      jint n_bins,
      jint n_frames) {
    if (data == nullptr) {
      return static_cast<jint>(Error::EndOfMethod);
    }
    auto data_size = data->size();
    if (data_size != 0) {
      std::vector<jfloat> data_jfloat(data_size);
      std::vector<float> data_f(data_size);
      data->getRegion(0, data_size, data_jfloat.data());
      for (int i = 0; i < data_size; i++) {
        data_f[i] = data_jfloat[i];
      }
      llm::Audio audio{std::move(data_f), batch_size, n_bins, n_frames};
      prefill_inputs_.emplace_back(llm::MultimodalInput{std::move(audio)});
    }
    return 0;
  }

  // Returns status_code
  jint append_raw_audio_input(
      facebook::jni::alias_ref<jbyteArray> data,
      jint batch_size,
      jint n_channels,
      jint n_samples) {
    if (data == nullptr) {
      return static_cast<jint>(Error::EndOfMethod);
    }
    auto data_size = data->size();
    if (data_size != 0) {
      std::vector<jbyte> data_jbyte(data_size);
      std::vector<uint8_t> data_u8(data_size);
      data->getRegion(0, data_size, data_jbyte.data());
      for (int i = 0; i < data_size; i++) {
        data_u8[i] = data_jbyte[i];
      }
      llm::RawAudio audio{
          std::move(data_u8), batch_size, n_channels, n_samples};
      prefill_inputs_.emplace_back(llm::MultimodalInput{std::move(audio)});
    }
    return 0;
  }

  void stop() {
    if (model_type_category_ == MODEL_TYPE_CATEGORY_MULTIMODAL) {
      multi_modal_runner_->stop();
    } else if (model_type_category_ == MODEL_TYPE_CATEGORY_LLM) {
      runner_->stop();
#if defined(EXECUTORCH_BUILD_QNN_MULTIMODAL)
    } else if (model_type_category_ == MODEL_TYPE_QNN_MULTIMODAL) {
      qnn_multimodal_runner_->stop();
#endif
    }
  }

  void reset_context() {
    if (runner_ != nullptr) {
      runner_->reset();
    }
    if (multi_modal_runner_ != nullptr) {
      multi_modal_runner_->reset();
    }
#if defined(EXECUTORCH_BUILD_QNN_MULTIMODAL)
    if (qnn_multimodal_runner_ != nullptr) {
      qnn_multimodal_runner_->reset();
    }
#endif
  }

  jint load() {
    int result = -1;
    std::stringstream ss;

    if (model_type_category_ == MODEL_TYPE_CATEGORY_MULTIMODAL) {
      result = static_cast<jint>(multi_modal_runner_->load());
      if (result != 0) {
        ss << "Failed to load multimodal runner: [" << result << "]";
      }
    } else if (model_type_category_ == MODEL_TYPE_CATEGORY_LLM) {
      result = static_cast<jint>(runner_->load());
      if (result != 0) {
        ss << "Failed to load llm runner: [" << result << "]";
      }
#if defined(EXECUTORCH_BUILD_QNN_MULTIMODAL)
    } else if (model_type_category_ == MODEL_TYPE_QNN_MULTIMODAL) {
      result = 0; // Start with success
      // Load encoder first
      if (encoder_runner_ != nullptr) {
        ET_LOG(Info, "Loading encoder runner...");
        auto encoder_result = encoder_runner_->load();
        if (encoder_result != executorch::runtime::Error::Ok) {
          ss << "Failed to load encoder runner";
          result = static_cast<jint>(encoder_result);
          ET_LOG(Error, "Encoder runner load failed: %d", result);
        } else {
          ET_LOG(Info, "Encoder runner loaded successfully");
        }
      } else {
        ET_LOG(Info, "Encoder runner is null, skipping encoder load");
      }
      // Then load the multimodal runner (only if encoder loaded successfully or was null)
      if (result == 0 && qnn_multimodal_runner_ != nullptr) {
        ET_LOG(Info, "Loading QNN multimodal runner...");
        result = static_cast<jint>(qnn_multimodal_runner_->load());
        if (result != 0) {
          ss << "Failed to load QNN multimodal runner: [" << result << "]";
          ET_LOG(Error, "QNN multimodal runner load failed: %d", result);
        } else {
          ET_LOG(Info, "QNN multimodal runner loaded successfully");
        }
      } else if (qnn_multimodal_runner_ == nullptr) {
        ss << "QNN multimodal runner is null";
        result = static_cast<jint>(Error::InvalidState);
        ET_LOG(Error, "QNN multimodal runner is null");
      }
#endif
    } else {
      ss << "Invalid model type category: " << model_type_category_
         << ". Valid values are: " << MODEL_TYPE_CATEGORY_LLM << ", "
         << MODEL_TYPE_CATEGORY_MULTIMODAL << ", or " << MODEL_TYPE_QNN_MULTIMODAL;
    }
    if (result != 0) {
      executorch::jni_helper::throwExecutorchException(
          static_cast<uint32_t>(Error::InvalidArgument), ss.str().c_str());
    }
    return result; // 0 on success to keep backward compatibility
  }

  static void registerNatives() {
    registerHybrid({
        makeNativeMethod("initHybrid", ExecuTorchLlmJni::initHybrid),
        makeNativeMethod("generate", ExecuTorchLlmJni::generate),
        makeNativeMethod("stop", ExecuTorchLlmJni::stop),
        makeNativeMethod("load", ExecuTorchLlmJni::load),
        makeNativeMethod(
            "appendImagesInput", ExecuTorchLlmJni::append_images_input),
        makeNativeMethod(
            "appendNormalizedImagesInput",
            ExecuTorchLlmJni::append_normalized_images_input),
        makeNativeMethod(
            "appendAudioInput", ExecuTorchLlmJni::append_audio_input),
        makeNativeMethod(
            "appendAudioInputFloat",
            ExecuTorchLlmJni::append_audio_input_float),
        makeNativeMethod(
            "appendRawAudioInput", ExecuTorchLlmJni::append_raw_audio_input),
        makeNativeMethod(
            "appendTextInput", ExecuTorchLlmJni::append_text_input),
        makeNativeMethod("resetContext", ExecuTorchLlmJni::reset_context),
#if defined(EXECUTORCH_BUILD_QNN_MULTIMODAL)
        makeNativeMethod(
            "encodeImageFromFileNative", ExecuTorchLlmJni::encode_image_from_file),
        makeNativeMethod(
            "encodeImageNative", ExecuTorchLlmJni::encode_image),
#endif
    });
  }
};

} // namespace executorch_jni

void register_natives_for_llm() {
  executorch_jni::ExecuTorchLlmJni::registerNatives();
}

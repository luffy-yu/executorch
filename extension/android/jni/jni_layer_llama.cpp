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

#include <android/log.h>

#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

// Helper macro for Android logging with ETLogging tag
#define ALOG(...) __android_log_print(ANDROID_LOG_DEBUG, "ETLogging", __VA_ARGS__)

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

#if defined(EXECUTORCH_BUILD_QNN)
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
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

 public:
  constexpr static auto kJavaDescriptor =
      "Lorg/pytorch/executorch/extension/llm/LlmModule;";

  constexpr static int MODEL_TYPE_CATEGORY_LLM = 1;
  constexpr static int MODEL_TYPE_CATEGORY_MULTIMODAL = 2;
  constexpr static int MODEL_TYPE_MEDIATEK_LLAMA = 3;
  constexpr static int MODEL_TYPE_QNN_LLAMA = 4;

  static facebook::jni::local_ref<jhybriddata> initHybrid(
      facebook::jni::alias_ref<jclass>,
      jint model_type_category,
      facebook::jni::alias_ref<jstring> model_path,
      facebook::jni::alias_ref<jstring> tokenizer_path,
      jfloat temperature,
      facebook::jni::alias_ref<jstring> data_path) {
    return makeCxxInstance(
        model_type_category,
        model_path,
        tokenizer_path,
        temperature,
        data_path);
  }

  ExecuTorchLlmJni(
      jint model_type_category,
      facebook::jni::alias_ref<jstring> model_path,
      facebook::jni::alias_ref<jstring> tokenizer_path,
      jfloat temperature,
      facebook::jni::alias_ref<jstring> data_path = nullptr) {
    // Force log at constructor start
    __android_log_print(ANDROID_LOG_FATAL, "ZZTEST", "=== CONSTRUCTOR START, category=%d ===", model_type_category);
    
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
    ALOG("[JNI] Model type category set to: %d", model_type_category_);
    ALOG("[JNI] Checking branches: MULTIMODM=%d, LLM=%d, QNN=%d, MEDIATEK=%d", 
           MODEL_TYPE_CATEGORY_MULTIMODAL, MODEL_TYPE_CATEGORY_LLM, 
           MODEL_TYPE_QNN_LLAMA, MODEL_TYPE_MEDIATEK_LLAMA);
    
    if (model_type_category == MODEL_TYPE_CATEGORY_MULTIMODAL) {
      ALOG("[JNI] Taking MULTIMODAL branch");
      multi_modal_runner_ = llm::create_multimodal_runner(
          model_path->toStdString().c_str(),
          llm::load_tokenizer(tokenizer_path->toStdString()));
    } else if (model_type_category == MODEL_TYPE_CATEGORY_LLM) {
      ALOG("[JNI] Taking LLM branch");
      std::optional<const std::string> data_path_str = data_path
          ? std::optional<const std::string>{data_path->toStdString()}
          : std::nullopt;
      if (data_path_str.has_value()) {
        ALOG("[JNI] data_path: %s", data_path_str.value().c_str());
      } else {
        ALOG("[JNI] data_path is null");
      }
      ALOG("[JNI] Creating text LLM runner");
      runner_ = executorch::extension::llm::create_text_llm_runner(
          model_path->toStdString(),
          llm::load_tokenizer(tokenizer_path->toStdString()),
          data_path_str);
#if defined(EXECUTORCH_BUILD_QNN)
    } else if (model_type_category == MODEL_TYPE_QNN_LLAMA) {
      ALOG("[QNN] Initializing QNN Llama model");
      ALOG("[QNN] Model path: %s", model_path->toStdString().c_str());
      ALOG("[QNN] Tokenizer path: %s", tokenizer_path->toStdString().c_str());
      ALOG("[QNN] Temperature: %f", temperature_);
      
      // Default configuration values
      std::string decoder_model = "qwen3";
      std::string kv_updater = "SmartMask";
      int eval_mode = 1; // kHybrid mode
      
      // Parse configuration from data_path if provided
      // Expected format: decoder_model_version:qwen3;kv_updater:SmartMask;eval_mode:1
      if (data_path != nullptr) {
        std::string config_str = data_path->toStdString();
        ALOG("[QNN] Parsing config string: %s", config_str.c_str());
        
        // Parse configuration
        size_t pos = 0;
        while (pos < config_str.length()) {
          size_t colon = config_str.find(':', pos);
          size_t semicolon = config_str.find(';', pos);
          
          if (colon != std::string::npos) {
            std::string key = config_str.substr(pos, colon - pos);
            std::string value;
            
            if (semicolon != std::string::npos && semicolon > colon) {
              value = config_str.substr(colon + 1, semicolon - colon - 1);
              pos = semicolon + 1;
            } else {
              value = config_str.substr(colon + 1);
              pos = config_str.length();
            }
            
            if (key == "decoder_model_version") {
              decoder_model = value;
              ALOG("[QNN] Using decoder_model_version: %s", decoder_model.c_str());
            } else if (key == "kv_updater") {
              kv_updater = value;
              ALOG("[QNN] Using kv_updater: %s", kv_updater.c_str());
            } else if (key == "eval_mode") {
              eval_mode = std::stoi(value);
              ALOG("[QNN] Using eval_mode: %d", eval_mode);
            }
          } else {
            break;
          }
        }
      } else {
        ALOG("[QNN] No config provided, using defaults");
      }
      
      ALOG("[QNN] Final config - decoder_model: %s, kv_updater: %s, eval_mode: %d", 
             decoder_model.c_str(), kv_updater.c_str(), eval_mode);
      
      std::unique_ptr<executorch::extension::Module> module;
      try {
        ALOG("[QNN] Creating Module without data files...");
        std::vector<std::string> empty_data_files;
        module = std::make_unique<executorch::extension::Module>(
            model_path->toStdString().c_str(),
            empty_data_files,
            executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
        ALOG("[QNN] Module created successfully");
      } catch (const std::exception& e) {
        ALOG("[QNN] FAILED to create Module: %s", e.what());
        throw;
      }
      
      ALOG("[QNN] Detecting kv_bitwidth...");
      example::KvBitWidth kv_bitwidth = example::KvBitWidth::kWidth8;
      if (module->method_names()->count("get_kv_io_bit_width") > 0) {
        kv_bitwidth = static_cast<example::KvBitWidth>(
            module->get("get_kv_io_bit_width").get().toScalar().to<int64_t>());
        ALOG("[QNN] Detected kv_bitwidth from model: %d", static_cast<int>(kv_bitwidth));
      } else {
        ALOG("[QNN] Using default kv_bitwidth: %d", static_cast<int>(kv_bitwidth));
      }

      if (kv_bitwidth == example::KvBitWidth::kWidth8) {
        ALOG("[QNN] Creating Runner<uint8_t> with:");
        ALOG("[QNN]   - decoder_model: %s", decoder_model.c_str());
        ALOG("[QNN]   - kv_updater: %s", kv_updater.c_str());
        ALOG("[QNN]   - eval_mode: %d", eval_mode);
        try {
          runner_ = std::make_unique<example::Runner<uint8_t>>(
              std::move(module),
              decoder_model, // const std::string&
              model_path->toStdString(), // const std::string&
              tokenizer_path->toStdString(), // const std::string&
              "", // performance_output_path (param 5)
              "", // dump_logits_path (param 6)
              temperature_, // temperature
              eval_mode, // eval_mode
              kv_updater // kv_updater
          );
          ALOG("[QNN] Runner<uint8_t> created successfully");
        } catch (const std::exception& e) {
          ALOG("[QNN] FAILED to create Runner<uint8_t>: %s", e.what());
          throw;
        }
      } else if (kv_bitwidth == example::KvBitWidth::kWidth16) {
        ALOG("[QNN] Creating Runner<uint16_t> with:");
        ALOG("[QNN]   - decoder_model: %s", decoder_model.c_str());
        ALOG("[QNN]   - kv_updater: %s", kv_updater.c_str());
        ALOG("[QNN]   - eval_mode: %d", eval_mode);
        try {
          runner_ = std::make_unique<example::Runner<uint16_t>>(
              std::move(module),
              decoder_model, // const std::string&
              model_path->toStdString(), // const std::string&
              tokenizer_path->toStdString(), // const std::string&
              "", // performance_output_path (param 5)
              "", // dump_logits_path (param 6)
              temperature_, // temperature
              eval_mode, // eval_mode
              kv_updater // kv_updater
          );
          ALOG("[QNN] Runner<uint16_t> created successfully");
        } catch (const std::exception& e) {
          ALOG("[QNN] FAILED to create Runner<uint16_t>: %s", e.what());
          throw;
        }
      } else {
        ET_CHECK_MSG(
            false,
            "Unsupported kv bitwidth: %ld",
            static_cast<int64_t>(kv_bitwidth));
      }
      model_type_category_ = MODEL_TYPE_CATEGORY_LLM;
      ALOG("[QNN] QNN initialization complete");
#endif
#if defined(EXECUTORCH_BUILD_MEDIATEK)
    } else if (model_type_category == MODEL_TYPE_MEDIATEK_LLAMA) {
      ALOG("[JNI] Taking MEDIATEK branch");
      runner_ = std::make_unique<MTKLlamaRunner>(
          model_path->toStdString().c_str(),
          tokenizer_path->toStdString().c_str());
      // Interpret the model type as LLM
      model_type_category_ = MODEL_TYPE_CATEGORY_LLM;
#endif
    } else {
      ALOG("[JNI] ERROR: Unknown model_type_category: %d", model_type_category);
      ALOG("[JNI] QNN support compiled: %s", 
#if defined(EXECUTORCH_BUILD_QNN)
        "YES"
#else
        "NO - QNN backend not included in this build!"
#endif
      );
      ALOG("[JNI] MediaTek support compiled: %s",
#if defined(EXECUTORCH_BUILD_MEDIATEK)
        "YES"
#else
        "NO"
#endif
      );
      ET_CHECK_MSG(false, "Unsupported model_type_category: %d. Check if backend is compiled in AAR.", model_type_category);
    }
    __android_log_print(ANDROID_LOG_FATAL, "ZZTEST", "=== CONSTRUCTOR END ===");
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
    }
  }

  void reset_context() {
    if (runner_ != nullptr) {
      runner_->reset();
    }
    if (multi_modal_runner_ != nullptr) {
      multi_modal_runner_->reset();
    }
  }

  jint load() {
    // Force a log that MUST appear
    __android_log_print(ANDROID_LOG_FATAL, "ZZTEST", "=== LOAD METHOD CALLED ===");
    __android_log_print(ANDROID_LOG_ERROR, "ExecuTorch", "[Load] Starting model load, category: %d", model_type_category_);
    ET_LOG(Error, "[Load] Starting model load, category: %d", model_type_category_);
    if (model_type_category_ == MODEL_TYPE_CATEGORY_MULTIMODAL) {
      ET_LOG(Error, "[Load] Loading multimodal model...");
      Error result = multi_modal_runner_->load();
      ET_LOG(Error, "[Load] Multimodal load result: %d", static_cast<int>(result));
      if (result != Error::Ok) {
        ET_LOG(Error, "[Load] Multimodal load FAILED with error code: %d", static_cast<int>(result));
      }
      return static_cast<jint>(result);
    } else if (model_type_category_ == MODEL_TYPE_CATEGORY_LLM) {
      __android_log_print(ANDROID_LOG_ERROR, "ExecuTorch", "[Load] Loading LLM model...");
      ET_LOG(Error, "[Load] Loading LLM model...");
      if (runner_ == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "ExecuTorch", "[Load] ERROR: Runner is null!");
        ET_LOG(Error, "[Load] ERROR: Runner is null!");
        return static_cast<jint>(Error::InvalidArgument);
      }
      __android_log_print(ANDROID_LOG_ERROR, "ExecuTorch", "[Load] Calling runner_->load()...");
      Error result = runner_->load();
      __android_log_print(ANDROID_LOG_ERROR, "ExecuTorch", "[Load] runner_->load() returned: %d", static_cast<int>(result));
      ET_LOG(Error, "[Load] LLM load result: %d", static_cast<int>(result));
      if (result != Error::Ok) {
        ET_LOG(Error, "[Load] LLM load FAILED with error code: %d", static_cast<int>(result));
      } else {
        ET_LOG(Error, "[Load] Model loaded successfully!");
      }
      return static_cast<jint>(result);
    }
    ET_LOG(Error, "[Load] ERROR: Invalid model category: %d", model_type_category_);
    return static_cast<jint>(Error::InvalidArgument);
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
    });
  }
};

} // namespace executorch_jni

void register_natives_for_llm() {
  executorch_jni::ExecuTorchLlmJni::registerNatives();
}

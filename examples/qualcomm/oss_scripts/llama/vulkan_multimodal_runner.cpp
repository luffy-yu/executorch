/*
 * Vulkan Multimodal Runner for FastVLM
 *
 * A self-contained multimodal VLM runner that uses ExecuTorch Vulkan backend.
 * No QNN dependency. Loads 3 .pte files:
 *   1. Vision encoder (image -> hidden states)
 *   2. Text embedding (token IDs -> embeddings)
 *   3. Text decoder (embeddings -> logits, with KV cache)
 *
 * The decoder takes pre-computed embeddings (NOT token IDs) as input,
 * allowing image features to be injected at placeholder positions.
 *
 * Decoder forward: (h: [1, 1, dim], input_pos: [1]) -> logits: [1, vocab_size]
 *
 * Usage:
 *   vulkan_multimodal_runner \
 *     --encoder_path vision_encoder.pte \
 *     --embedding_path text_embedding.pte \
 *     --decoder_path decoder.pte \
 *     --tokenizer_path tokenizer.json \
 *     --image_path image.jpg \
 *     --prompt "can you describe this image" \
 *     --seq_len 128
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/encoder.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::extension::llm::time_in_ms;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;

// ============================================================================
// Command-line flags
// ============================================================================
DEFINE_string(encoder_path, "", "Path to vision encoder .pte file");
DEFINE_string(embedding_path, "", "Path to text embedding .pte file");
DEFINE_string(decoder_path, "", "Path to text decoder .pte file");
DEFINE_string(tokenizer_path, "", "Path to tokenizer.json");
DEFINE_string(image_path, "", "Path to input image (JPG/PNG/BMP)");
DEFINE_string(prompt, "Describe this image:", "Text prompt");
DEFINE_string(
    output_path,
    "outputs.txt",
    "Output file for generated text");
DEFINE_int32(seq_len, 128, "Maximum sequence length to generate");
DEFINE_double(temperature, 0.0, "Sampling temperature (0 = greedy)");
DEFINE_bool(encode_only, false, "Only run encode stage, save embeddings to file");
DEFINE_bool(decode_only, false, "Only run decode stage, load embeddings from file");
DEFINE_string(
    embeddings_file,
    "/data/local/tmp/fastvlm/merged_embeddings.bin",
    "Path to save/load merged embeddings binary file");

// ============================================================================
// Helper: format prompt for FastVLM (Qwen2-based chat template)
// ============================================================================
static std::string format_fastvlm_prompt(
    const std::string& prompt,
    int image_seq_len) {
  std::string formatted;
  formatted += "<|im_start|>system\n";
  formatted += "You are a helpful assistant.";
  formatted += "<|im_end|>\n";
  formatted += "<|im_start|>user\n";
  // Insert image tokens
  for (int i = 0; i < image_seq_len; ++i) {
    formatted += "<image>";
  }
  formatted += prompt;
  formatted += "<|im_end|>\n<|im_start|>assistant\n";
  return formatted;
}

// ============================================================================
// Helper: replace tokenized <image> sequences with placeholder token (-200)
// The Qwen tokenizer encodes <image> as multiple sub-tokens.
// ============================================================================
static std::vector<uint64_t> replace_image_tokens(
    const std::vector<uint64_t>& tokens) {
  const uint64_t TOKEN_LT = 27;      // "<"
  const uint64_t TOKEN_IMAGE = 1805;  // "image"
  const uint64_t TOKEN_GT = 29;       // ">"
  const uint64_t TOKEN_GT_LT = 1784;  // "><"
  const uint64_t TOKEN_GT_NL = 397;   // ">\n"
  const int64_t PLACEHOLDER = -200;

  std::vector<uint64_t> result;
  size_t i = 0;
  while (i < tokens.size()) {
    if (tokens[i] == TOKEN_LT && i + 1 < tokens.size() &&
        tokens[i + 1] == TOKEN_IMAGE) {
      result.push_back(static_cast<uint64_t>(PLACEHOLDER));
      i += 2;
      while (i < tokens.size()) {
        if (tokens[i] == TOKEN_GT || tokens[i] == TOKEN_GT_NL) {
          i += 1;
          break;
        } else if (tokens[i] == TOKEN_GT_LT) {
          result.push_back(static_cast<uint64_t>(PLACEHOLDER));
          i += 1;
          if (i < tokens.size() && tokens[i] == TOKEN_IMAGE) {
            i += 1;
          } else {
            break;
          }
        } else {
          break;
        }
      }
    } else {
      result.push_back(tokens[i]);
      i += 1;
    }
  }
  return result;
}

// ============================================================================
// Helper: greedy argmax over logits
// ============================================================================
static int32_t greedy_sample(const float* logits, int64_t vocab_size) {
  int32_t best = 0;
  float best_val = logits[0];
  for (int32_t v = 1; v < vocab_size; v++) {
    if (logits[v] > best_val) {
      best_val = logits[v];
      best = v;
    }
  }
  return best;
}

// ============================================================================
// Main Vulkan multimodal runner
// ============================================================================
int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  ET_LOG(Info, "=== Vulkan Multimodal Runner for FastVLM ===");
  ET_LOG(Info, "Encoder:   %s", FLAGS_encoder_path.c_str());
  ET_LOG(Info, "Embedding: %s", FLAGS_embedding_path.c_str());
  ET_LOG(Info, "Decoder:   %s", FLAGS_decoder_path.c_str());
  ET_LOG(Info, "Tokenizer: %s", FLAGS_tokenizer_path.c_str());
  ET_LOG(Info, "Image:     %s", FLAGS_image_path.c_str());
  ET_LOG(Info, "Prompt:    %s", FLAGS_prompt.c_str());
  ET_LOG(Info, "Seq len:   %d", FLAGS_seq_len);
  if (FLAGS_encode_only) {
    ET_LOG(Info, "Mode:      ENCODE ONLY (saving to %s)", FLAGS_embeddings_file.c_str());
  } else if (FLAGS_decode_only) {
    ET_LOG(Info, "Mode:      DECODE ONLY (loading from %s)", FLAGS_embeddings_file.c_str());
  }

  long total_start_ms = time_in_ms();

  // Variables used by both stages
  int32_t num_tokens = 0;
  int32_t embedding_dim = 896;  // Qwen2-0.5B
  std::vector<float> all_embeddings;
  std::vector<uint64_t> prompt_tokens;
  const int32_t EMBED_BATCH = 16;  // Embedding .pte fixed input shape [1, 16]

  // Embedding module - needed for encode (Step 3) and decode (generation loop)
  std::unique_ptr<Module> embedding_module;
  if (!FLAGS_embedding_path.empty()) {
    embedding_module = std::make_unique<Module>(
        FLAGS_embedding_path, Module::LoadMode::MmapUseMlockIgnoreErrors);
  }

  // ========================================================================
  // Decode-only mode: load embeddings from file, skip Steps 1-4
  // ========================================================================
  if (FLAGS_decode_only) {
    ET_LOG(Info, "");
    ET_LOG(Info, "[Decode-only] Loading merged embeddings from %s...",
           FLAGS_embeddings_file.c_str());

    std::ifstream emb_file(FLAGS_embeddings_file, std::ios::binary);
    ET_CHECK_MSG(emb_file.is_open(), "Failed to open embeddings file: %s",
                 FLAGS_embeddings_file.c_str());

    // Read header: num_tokens, embedding_dim
    emb_file.read(reinterpret_cast<char*>(&num_tokens), sizeof(int32_t));
    emb_file.read(reinterpret_cast<char*>(&embedding_dim), sizeof(int32_t));
    ET_LOG(Info, "  Header: num_tokens=%d, embedding_dim=%d", num_tokens, embedding_dim);

    // Read embeddings
    all_embeddings.resize(num_tokens * embedding_dim);
    emb_file.read(reinterpret_cast<char*>(all_embeddings.data()),
                  num_tokens * embedding_dim * sizeof(float));

    // Read token IDs (for reference, not used in decode)
    int32_t num_tok_ids = 0;
    emb_file.read(reinterpret_cast<char*>(&num_tok_ids), sizeof(int32_t));
    prompt_tokens.resize(num_tok_ids);
    emb_file.read(reinterpret_cast<char*>(prompt_tokens.data()),
                  num_tok_ids * sizeof(uint64_t));
    emb_file.close();

    ET_LOG(Info, "  Loaded %d embeddings (%.1f MB)",
           num_tokens, num_tokens * embedding_dim * sizeof(float) / (1024.0 * 1024.0));
  }

  if (!FLAGS_decode_only) {
  // ========================================================================
  // Step 1: Load vision encoder and encode image
  // ========================================================================
  ET_LOG(Info, "");
  ET_LOG(Info, "[Step 1] Loading vision encoder and encoding image...");

  auto encoder_runner =
      std::make_unique<example::EncoderRunner>(FLAGS_encoder_path);
  ET_CHECK_MSG(
      encoder_runner->load() == Error::Ok, "Failed to load encoder");

  auto encode_result =
      encoder_runner->encode_from_file(FLAGS_image_path);
  ET_CHECK_MSG(encode_result.ok(), "Failed to encode image");

  auto image_hidden_states = encode_result.get();
  int32_t image_seq_len = encoder_runner->get_image_seq_len();
  embedding_dim = image_hidden_states.size(2);

  ET_LOG(
      Info,
      "  Image encoded: shape=[1, %d, %d]",
      image_seq_len,
      embedding_dim);

  // ========================================================================
  // Step 2: Load tokenizer and tokenize prompt
  // ========================================================================
  ET_LOG(Info, "");
  ET_LOG(Info, "[Step 2] Loading tokenizer and formatting prompt...");

  auto tokenizer = std::make_unique<tokenizers::HFTokenizer>();
  auto tok_result = tokenizer->load(FLAGS_tokenizer_path);
  ET_CHECK_MSG(tok_result == tokenizers::Error::Ok, "Failed to load tokenizer");

  // Format prompt with chat template and image tokens
  std::string formatted_prompt =
      format_fastvlm_prompt(FLAGS_prompt, image_seq_len);
  ET_LOG(Info, "  Formatted prompt (length=%zu)", formatted_prompt.size());

  // Tokenize (n_bos=0 for Qwen2)
  auto encode_tok = tokenizer->encode(formatted_prompt, 0, 0);
  ET_CHECK_MSG(encode_tok.ok(), "Failed to tokenize prompt");
  prompt_tokens = encode_tok.get();

  ET_LOG(Info, "  Tokenized: %zu tokens", prompt_tokens.size());

  // Replace <image> sub-tokens with placeholder -200
  prompt_tokens = replace_image_tokens(prompt_tokens);
  ET_LOG(
      Info,
      "  After image token replacement: %zu tokens",
      prompt_tokens.size());

  // Count placeholder tokens
  int placeholder_count = 0;
  for (auto t : prompt_tokens) {
    if (static_cast<int64_t>(t) == -200)
      placeholder_count++;
  }
  ET_LOG(Info, "  Placeholder tokens (-200): %d", placeholder_count);
  ET_LOG(Info, "  Image seq len: %d", image_seq_len);

  num_tokens = static_cast<int32_t>(prompt_tokens.size());

  // ========================================================================
  // Step 3: Load text embedding and generate embeddings
  // ========================================================================
  ET_LOG(Info, "");
  ET_LOG(Info, "[Step 3] Loading text embedding model...");

  // embedding_module already loaded at top scope

  // Embed tokens one at a time (the embedding .pte has fixed input shape)
  // For non-image tokens: embed via embedding_module
  // For image tokens (-200): we'll replace with image features later
  ET_LOG(Info, "  Embedding %d tokens one at a time...", num_tokens);
  long embed_start_ms = time_in_ms();

  // First, get embedding_dim from the encoder output
  // We already know it: embedding_dim from step 1
  // Allocate buffer for all embeddings: [num_tokens, embedding_dim]
  all_embeddings.resize(num_tokens * embedding_dim, 0.0f);

  // The embedding .pte has fixed input shape (1, 16), so we batch tokens
  // in groups of EMBED_BATCH and extract the relevant embeddings.

  // Collect non-image token indices and IDs
  std::vector<int32_t> text_positions;
  std::vector<int64_t> text_token_ids;
  for (int32_t i = 0; i < num_tokens; i++) {
    int64_t tok = static_cast<int64_t>(prompt_tokens[i]);
    if (tok != -200) {
      text_positions.push_back(i);
      text_token_ids.push_back(tok);
    }
  }

  ET_LOG(Info, "  Embedding %zu text tokens in batches of %d...",
         text_positions.size(), EMBED_BATCH);

  // Process in batches of EMBED_BATCH
  for (size_t batch_start = 0; batch_start < text_token_ids.size();
       batch_start += EMBED_BATCH) {
    size_t batch_end =
        std::min(batch_start + EMBED_BATCH, text_token_ids.size());
    size_t actual_batch = batch_end - batch_start;

    // Prepare padded input [1, EMBED_BATCH]
    std::vector<int64_t> batch_tokens(EMBED_BATCH, 0);
    for (size_t j = 0; j < actual_batch; j++) {
      batch_tokens[j] = text_token_ids[batch_start + j];
    }

    std::vector<int32_t> tok_sizes = {1, EMBED_BATCH};
    auto tok_tensor = executorch::extension::from_blob(
        batch_tokens.data(), tok_sizes, ScalarType::Long);

    std::vector<EValue> emb_inputs;
    emb_inputs.emplace_back(*tok_tensor.get());
    auto emb_result = embedding_module->forward(emb_inputs);
    ET_CHECK_MSG(
        emb_result.ok(),
        "Embedding failed for batch starting at %zu",
        batch_start);

    auto emb_tensor = emb_result.get()[0].toTensor();
    const float* emb_data = emb_tensor.const_data_ptr<float>();

    // Copy embeddings to the correct positions
    for (size_t j = 0; j < actual_batch; j++) {
      int32_t pos = text_positions[batch_start + j];
      std::memcpy(
          all_embeddings.data() + pos * embedding_dim,
          emb_data + j * embedding_dim,
          embedding_dim * sizeof(float));
    }
  }

  long embed_end_ms = time_in_ms();
  ET_LOG(
      Info,
      "  Text embedding: %ld ms (%.1f ms/token)",
      embed_end_ms - embed_start_ms,
      static_cast<double>(embed_end_ms - embed_start_ms) / num_tokens);

  // ========================================================================
  // Step 4: Merge text embeddings with image hidden states
  // ========================================================================
  ET_LOG(Info, "");
  ET_LOG(Info, "[Step 4] Merging text + image embeddings...");

  // all_embeddings already has text embeddings; now replace image placeholders
  const float* image_data = image_hidden_states.const_data_ptr<float>();
  int img_idx = 0;
  for (size_t i = 0; i < prompt_tokens.size(); ++i) {
    if (static_cast<int64_t>(prompt_tokens[i]) == -200 &&
        img_idx < image_seq_len) {
      std::memcpy(
          all_embeddings.data() + i * embedding_dim,
          image_data + img_idx * embedding_dim,
          embedding_dim * sizeof(float));
      img_idx++;
    }
  }
  // all_embeddings now contains merged text + image embeddings
  ET_LOG(
      Info,
      "  Merged %d image embeddings into %d token positions",
      img_idx,
      num_tokens);

  // ========================================================================
  // Encode-only mode: save merged embeddings to file and exit
  // ========================================================================
  if (FLAGS_encode_only) {
    ET_LOG(Info, "");
    ET_LOG(Info, "[Encode-only] Saving merged embeddings to %s...",
           FLAGS_embeddings_file.c_str());

    std::ofstream emb_file(FLAGS_embeddings_file, std::ios::binary);
    ET_CHECK_MSG(emb_file.is_open(), "Failed to open output file: %s",
                 FLAGS_embeddings_file.c_str());

    // Write header: num_tokens, embedding_dim
    emb_file.write(reinterpret_cast<const char*>(&num_tokens), sizeof(int32_t));
    emb_file.write(reinterpret_cast<const char*>(&embedding_dim), sizeof(int32_t));

    // Write embeddings
    emb_file.write(reinterpret_cast<const char*>(all_embeddings.data()),
                   num_tokens * embedding_dim * sizeof(float));

    // Write token IDs (for reference)
    int32_t num_tok_ids = static_cast<int32_t>(prompt_tokens.size());
    emb_file.write(reinterpret_cast<const char*>(&num_tok_ids), sizeof(int32_t));
    emb_file.write(reinterpret_cast<const char*>(prompt_tokens.data()),
                   num_tok_ids * sizeof(uint64_t));
    emb_file.close();

    float size_mb = num_tokens * embedding_dim * sizeof(float) / (1024.0f * 1024.0f);
    long total_end_ms = time_in_ms();
    ET_LOG(Info, "  Saved %d embeddings (%.1f MB) in %ld ms",
           num_tokens, size_mb, total_end_ms - total_start_ms);
    ET_LOG(Info, "Encode stage complete!");
    return 0;
  }

  } // end if (!FLAGS_decode_only)

  // ========================================================================
  // Step 5: Load decoder and run inference
  // ========================================================================
  ET_LOG(Info, "");
  ET_LOG(Info, "[Step 5] Loading decoder and running inference...");

  // Load tokenizer for decode stage
  auto decode_tokenizer = std::make_unique<tokenizers::HFTokenizer>();
  {
    auto tok_result = decode_tokenizer->load(FLAGS_tokenizer_path);
    ET_CHECK_MSG(tok_result == tokenizers::Error::Ok, "Failed to load tokenizer for decode");
  }
  auto* tokenizer_ptr = decode_tokenizer.get();

  auto decoder_module = std::make_unique<Module>(
      FLAGS_decoder_path, Module::LoadMode::MmapUseMlockIgnoreErrors);

  // The decoder takes (h: [1, 1, dim], input_pos: [1]) -> logits
  // We need to feed each token position one-by-one to build the KV cache

  // Setup vocab size (Qwen2)
  const int64_t vocab_size = 151936;

  // Get EOS token IDs
  std::unordered_set<uint64_t> eos_ids;
  auto eos_result = tokenizer_ptr->encode("<|im_end|>", 0, 0);
  if (eos_result.ok()) {
    for (auto id : eos_result.get()) {
      eos_ids.insert(id);
    }
  }
  auto eos_result2 = tokenizer_ptr->encode("<|endoftext|>", 0, 0);
  if (eos_result2.ok()) {
    for (auto id : eos_result2.get()) {
      eos_ids.insert(id);
    }
  }

  ET_LOG(Info, "  EOS token IDs: ");
  for (auto id : eos_ids) {
    ET_LOG(Info, "    %lu", id);
  }

  // ========================================================================
  // Prefill: feed each merged embedding through the decoder
  // ========================================================================
  ET_LOG(Info, "");
  ET_LOG(Info, "  Prefilling %d tokens through decoder...", num_tokens);
  long prefill_start_ms = time_in_ms();

  int32_t last_logit_token = -1;
  std::vector<float> last_logits_buf(vocab_size);

  for (int32_t pos = 0; pos < num_tokens; pos++) {
    // Extract embedding for this position: [1, 1, dim]
    std::vector<float> pos_embedding(embedding_dim);
    std::memcpy(
        pos_embedding.data(),
        all_embeddings.data() + pos * embedding_dim,
        embedding_dim * sizeof(float));

    std::vector<int32_t> h_sizes = {1, 1, embedding_dim};
    auto h_tensor = executorch::extension::from_blob(
        pos_embedding.data(), h_sizes, ScalarType::Float);

    // input_pos for this position
    std::vector<int64_t> input_pos_data = {static_cast<int64_t>(pos)};
    std::vector<int32_t> pos_sizes = {1};
    auto pos_tensor = executorch::extension::from_blob(
        input_pos_data.data(), pos_sizes, ScalarType::Long);

    std::vector<EValue> dec_inputs;
    dec_inputs.emplace_back(*h_tensor.get());
    dec_inputs.emplace_back(*pos_tensor.get());

    auto dec_result = decoder_module->forward(dec_inputs);
    ET_CHECK_MSG(dec_result.ok(), "Decoder forward failed at pos %d", pos);

    // Keep track of logits from last prefill position
    if (pos == num_tokens - 1) {
      auto logits = dec_result.get()[0].toTensor();
      const float* logits_data = logits.const_data_ptr<float>();
      std::memcpy(
          last_logits_buf.data(),
          logits_data,
          vocab_size * sizeof(float));
    }

    if ((pos + 1) % 50 == 0 || pos == num_tokens - 1) {
      ET_LOG(Info, "    Prefilled %d/%d tokens", pos + 1, num_tokens);
    }
  }

  long prefill_end_ms = time_in_ms();
  ET_LOG(
      Info,
      "  Prefill complete: %ld ms (%.1f ms/token)",
      prefill_end_ms - prefill_start_ms,
      static_cast<double>(prefill_end_ms - prefill_start_ms) / num_tokens);

  // ========================================================================
  // Generate: auto-regressive token generation
  // ========================================================================
  ET_LOG(Info, "");
  ET_LOG(Info, "  Generating tokens...");

  std::string generated_text;
  std::vector<uint64_t> generated_tokens;
  int max_gen_tokens = FLAGS_seq_len;

  // Sample first token from prefill logits
  uint64_t cur_token = greedy_sample(last_logits_buf.data(), vocab_size);
  auto first_decode = tokenizer_ptr->decode(0, cur_token);
  if (first_decode.ok()) {
    std::string piece = first_decode.get();
    generated_text += piece;
    printf("%s", piece.c_str());
    fflush(stdout);
  }
  generated_tokens.push_back(cur_token);

  int32_t cur_pos = num_tokens;
  uint64_t prev_token = cur_token;
  long gen_start_ms = time_in_ms();

  for (int gen_i = 1; gen_i < max_gen_tokens; gen_i++) {
    if (eos_ids.count(cur_token) > 0) {
      ET_LOG(Info, "\n  EOS token reached after %d tokens", gen_i);
      break;
    }

    // Embed current token - pad to EMBED_BATCH (embedding .pte expects [1, 16])
    std::vector<int64_t> cur_tok_data(EMBED_BATCH, 0);
    cur_tok_data[0] = static_cast<int64_t>(cur_token);
    std::vector<int32_t> cur_tok_sizes = {1, EMBED_BATCH};
    auto cur_tok_tensor = executorch::extension::from_blob(
        cur_tok_data.data(), cur_tok_sizes, ScalarType::Long);

    std::vector<EValue> emb_inputs;
    emb_inputs.emplace_back(*cur_tok_tensor.get());
    auto emb_result = embedding_module->forward(emb_inputs);
    ET_CHECK_MSG(emb_result.ok(), "Token embedding failed at gen step %d", gen_i);
    auto emb_batch = emb_result.get()[0].toTensor();

    // Extract only the first embedding [1, 1, dim] from [1, 16, dim]
    const float* emb_batch_data = emb_batch.const_data_ptr<float>();
    std::vector<float> single_emb(embedding_dim);
    std::memcpy(single_emb.data(), emb_batch_data, embedding_dim * sizeof(float));
    std::vector<int32_t> emb_sizes = {1, 1, static_cast<int32_t>(embedding_dim)};
    auto token_embedding = executorch::extension::from_blob(
        single_emb.data(), emb_sizes, ScalarType::Float);

    // Pass embedding through decoder
    // input_pos for this position
    std::vector<int64_t> input_pos_data = {static_cast<int64_t>(cur_pos)};
    std::vector<int32_t> pos_sizes = {1};
    auto pos_tensor = executorch::extension::from_blob(
        input_pos_data.data(), pos_sizes, ScalarType::Long);

    std::vector<EValue> dec_inputs;
    dec_inputs.emplace_back(*token_embedding.get());
    dec_inputs.emplace_back(*pos_tensor.get());

    auto dec_result = decoder_module->forward(dec_inputs);
    ET_CHECK_MSG(
        dec_result.ok(),
        "Decoder forward failed at gen step %d (pos %d)",
        gen_i,
        cur_pos);

    auto logits = dec_result.get()[0].toTensor();
    const float* logits_data = logits.const_data_ptr<float>();

    // Sample next token
    prev_token = cur_token;
    cur_token = greedy_sample(logits_data, vocab_size);

    auto decode_tok = tokenizer_ptr->decode(prev_token, cur_token);
    if (decode_tok.ok()) {
      std::string piece = decode_tok.get();
      generated_text += piece;
      printf("%s", piece.c_str());
      fflush(stdout);
    }
    generated_tokens.push_back(cur_token);
    cur_pos++;
  }

  long gen_end_ms = time_in_ms();
  int num_gen = generated_tokens.size();
  double ms_per_token =
      num_gen > 1
      ? static_cast<double>(gen_end_ms - gen_start_ms) / (num_gen - 1)
      : 0;

  printf("\n");
  ET_LOG(Info, "");
  ET_LOG(Info, "========================================");
  ET_LOG(Info, "Generation complete:");
  ET_LOG(Info, "  Prefill tokens: %d", num_tokens);
  ET_LOG(
      Info,
      "  Prefill time: %ld ms (%.1f ms/token)",
      prefill_end_ms - prefill_start_ms,
      static_cast<double>(prefill_end_ms - prefill_start_ms) / num_tokens);
  ET_LOG(Info, "  Generated tokens: %d", num_gen);
  ET_LOG(
      Info,
      "  Generation time: %ld ms (%.1f ms/token)",
      gen_end_ms - gen_start_ms,
      ms_per_token);
  ET_LOG(Info, "========================================");

  // Save output
  long total_end_ms = time_in_ms();
  ET_LOG(Info, "");
  ET_LOG(Info, "Total pipeline time: %ld ms", total_end_ms - total_start_ms);
  ET_LOG(Info, "Generated text: %s", generated_text.c_str());

  std::ofstream fout(FLAGS_output_path);
  if (fout.is_open()) {
    fout << generated_text;
    fout.close();
    ET_LOG(Info, "Output saved to: %s", FLAGS_output_path.c_str());
  }

  return 0;
}

# FastVLM Integration Guide for QNN Backend

This document describes the process of integrating FastVLM (a vision-language model) with the Qualcomm QNN backend in ExecuTorch. It includes the fixes made, lessons learned, and a step-by-step guide for future VLM integrations.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Integration Steps](#integration-steps)
4. [Issues Encountered and Fixes](#issues-encountered-and-fixes)
5. [Lessons Learned](#lessons-learned)
6. [Guide for Future VLM Integrations](#guide-for-future-vlm-integrations)
7. [Testing and Validation](#testing-and-validation)
8. [Performance Considerations](#performance-considerations)

---

## Overview

FastVLM is a vision-language model that uses FastViTHD as its vision encoder and Qwen2-0.5B as its text decoder. The integration enables running FastVLM on Qualcomm Snapdragon devices using the Hexagon HTP (Hexagon Tensor Processor).

### Model Components

| Component | Model | Input | Output |
|-----------|-------|-------|--------|
| Vision Encoder | FastViTHD | 1024x1024 RGB image | 256 x 896 embeddings |
| Text Embedding | Qwen2 Embedding | Token IDs | 896-dim embeddings |
| Text Decoder | Qwen2-0.5B | Merged embeddings | Next token logits |

---

## Architecture

```
                    ┌─────────────────┐
                    │  Input Image    │
                    │  (1024x1024)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Vision Encoder  │
                    │   (FastViTHD)   │
                    │  vision_encoder │
                    │    _qnn.pte     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Image Hidden    │
                    │ States (256x896)│
                    └────────┬────────┘
                             │
    ┌─────────────┐          │          ┌─────────────┐
    │ Text Prompt │          │          │   Merged    │
    │  (tokens)   │          │          │ Embeddings  │
    └──────┬──────┘          │          └──────┬──────┘
           │                 │                 │
    ┌──────▼──────┐          │          ┌──────▼──────┐
    │    Text     │          │          │    Text     │
    │  Embedding  ├──────────┴──────────►   Decoder   │
    │text_embedding                     │hybrid_llama │
    │  _qnn.pte   │                     │  _qnn.pte   │
    └─────────────┘                     └──────┬──────┘
                                               │
                                        ┌──────▼──────┐
                                        │   Output    │
                                        │   Tokens    │
                                        └─────────────┘
```

---

## Integration Steps

### 1. Model Configuration

Create a decoder model configuration in `decoder/model_config.py`:

```python
@dataclass
class FastVLM_0_5B(QwenModelConfig):
    decoder_model_version: str = "fastvlm_0_5b"
    instruct_model: bool = True
    # ... other Qwen2-0.5B parameters
```

### 2. Vision Encoder Implementation

Create the vision encoder wrapper in `model/vision_encoder.py`:

```python
class FastVLMVisionEncoder(nn.Module):
    def __init__(self, checkpoint_path: str = None, ...):
        # Load FastViTHD model
        # Apply reparameterization after loading weights
        self._load_from_checkpoint(checkpoint_path)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Process image through vision tower
        # Apply MLP projection
        return image_hidden_states
```

### 3. Encoder Configuration

Add encoder config in `encoder/encoder_config.py`:

```python
@dataclass(init=False, frozen=True)
class FastVLMEncoder(VisionModalityConfig):
    encoder_class = FastVLMVisionEncoder
    img_seq_len = 256  # 16x16 patches
    img_resized_h = 1024
    img_resized_w = 1024
    quant_recipe = None  # FP16 for accuracy (see lessons learned)
```

### 4. Quantization Recipe (Optional)

Create quantization recipe in `encoder/encoder_quant_recipe.py`:

```python
class FastVLM_Encoder_QuantRecipe(EncoderQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a8w

    def __init__(self, verbose: bool = False):
        self.recipe = QuantRecipe(
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_TENSOR,
            verbose=verbose,
        )
```

### 5. C++ Runtime Support

Add model version enum in `qnn_multimodal_runner.cpp`:

```cpp
enum class MultimodalDecoderModelVersion {
    kSmolvlm,
    kInternvl3,
    kFastvlm,  // Add new model
    kUnknown
};
```

Add prompt formatting:

```cpp
case example::MultimodalDecoderModelVersion::kFastvlm:
    // FastVLM uses Qwen2/ChatML format with default system prompt
    formatted_prompt.append("<|im_start|>system\n");
    if (!system_prompt.empty()) {
        formatted_prompt.append(system_prompt);
    } else {
        formatted_prompt.append("You are a helpful assistant.");
    }
    formatted_prompt.append("<|im_end|>\n");
    formatted_prompt.append("<|im_start|>user\n");
    formatted_prompt.append(specials.image_token);
    formatted_prompt.append(prompt);
    formatted_prompt.append("<|im_end|>\n<|im_start|>assistant\n");
    break;
```

---

## Issues Encountered and Fixes

### Issue 1: BatchNorm Layers with Tiny Variances

**Symptom**: Vision encoder output was ~50x larger than expected with poor correlation (0.19) between compile-time and runtime.

**Root Cause**: The FastViTHD model has ConvFFN modules containing Conv2d + BatchNorm2d layers. These BatchNorm layers had extremely small `running_var` values (as low as 1e-8), which:
- Are below FP16 precision limits (smallest normal: 6.1e-5)
- Cause numerical instability when computing `1/sqrt(var + eps)`
- Lead to very large scale factors (up to 2082x)

**Fix**: Added BatchNorm reparameterization (fusing Conv+BN) for ConvFFN modules:

```python
# In model/mci.py - ConvFFN class
def reparameterize(self) -> None:
    """Fuse Conv2d and BatchNorm2d into a single Conv2d."""
    conv = self.conv.conv
    bn = self.conv.bn

    # Compute fused parameters
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = bn.weight / std
    fused_weight = conv.weight * scale.view(-1, 1, 1, 1)
    fused_bias = bn.bias - bn.weight * bn.running_mean / std

    # Create fused conv
    fused_conv = nn.Conv2d(...)
    fused_conv.weight.data = fused_weight
    fused_conv.bias.data = fused_bias
    self.conv = fused_conv
```

**Critical**: Call reparameterization **AFTER** loading weights, not before:

```python
# In vision_encoder.py - _load_from_checkpoint()
def _load_from_checkpoint(self, checkpoint_path):
    # Load weights first
    self.vision_tower.load_state_dict(state_dict)

    # THEN reparameterize (uses the loaded running_mean/var)
    self._reparameterize_convffn()
```

### Issue 2: System Prompt Mismatch

**Symptom**: Model generated garbage text ("Klawt brand names...") despite correct vision encoder output. Token counts differed: compile=281, runtime=270.

**Root Cause**: The Qwen2 tokenizer's chat template automatically adds a default system prompt:

```jinja
{% if loop.first and messages[0]['role'] != 'system' %}
{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}
{% endif %}
```

The C++ runtime was not adding this default system prompt, causing a 11-token mismatch.

**Fix**: Updated C++ runner to include default system prompt for FastVLM:

```cpp
case example::MultimodalDecoderModelVersion::kFastvlm:
    // Always add system prompt to match tokenizer's chat template
    formatted_prompt.append("<|im_start|>system\n");
    if (!system_prompt.empty()) {
        formatted_prompt.append(system_prompt);
    } else {
        formatted_prompt.append("You are a helpful assistant.");
    }
    formatted_prompt.append("<|im_end|>\n");
    // ... rest of prompt
```

### Issue 3: Vision Encoder Quantization Accuracy Loss

**Symptom**: Even with BatchNorm fix, quantized (16a8w) vision encoder produced different output than compile-time model (correlation: 0.26).

**Root Cause**: The 8-bit weight quantization (16a8w) caused significant accuracy loss for the complex FastViTHD architecture with its many conv layers and attention mechanisms.

**Temporary Fix**: Disable quantization for vision encoder:

```python
# In encoder_config.py
class FastVLMEncoder(VisionModalityConfig):
    quant_recipe = None  # Use FP16 instead of quantized
```

**Future Improvements**:
- Try 16a16w (16-bit weights) quantization
- Use HistogramObserver instead of MinMaxObserver
- Use larger/more diverse calibration dataset
- Apply per-channel quantization for more layers

---

## Lessons Learned

### 1. BatchNorm Handling is Critical

**Lesson**: Always check for BatchNorm layers in vision encoders and reparameterize them before QNN export.

**Why**:
- BatchNorm `running_var` values can be extremely small after training
- Small variances cause numerical overflow when computing scale factors
- FP16 precision limits exacerbate this issue

**Checklist**:
- [ ] Identify all BatchNorm layers in the model
- [ ] Check `running_var` values (any < 1e-5 is problematic)
- [ ] Implement reparameterization for Conv+BN pairs
- [ ] Reparameterize AFTER loading weights, not before

### 2. Prompt Consistency is Essential

**Lesson**: The exact prompt format at runtime must match compile-time, including default system prompts.

**Why**:
- Tokenizers may inject default content (system prompts, special tokens)
- Position embeddings are calibrated for specific token positions
- Even small misalignments cause the model to generate garbage

**Checklist**:
- [ ] Review tokenizer's chat_template for any default injections
- [ ] Match C++ runtime prompt formatting exactly
- [ ] Compare token counts between compile and runtime
- [ ] Debug by saving and comparing embeddings

### 3. Quantization Requires Careful Validation

**Lesson**: Always validate quantized model accuracy before deployment.

**Why**:
- Vision encoders are often more sensitive to quantization than LLMs
- Complex architectures (attention, many conv layers) accumulate quantization errors
- Per-tensor quantization may not be sufficient for all layers

**Validation Steps**:
1. Save compile-time outputs (embeddings, hidden states)
2. Save runtime outputs
3. Compare correlation, max difference, mean difference
4. Correlation < 0.9 indicates a problem

### 4. Debug with Intermediate Outputs

**Lesson**: Add comprehensive debug logging for intermediate tensors.

**Implementation**:
```cpp
// In encoder.cpp
ET_LOG(Info, "[DEBUG] Runtime vision encoder output: numel=%zu, range=[%.4f, %.4f], mean=%.4f",
       output.numel(), output.min(), output.max(), output.mean());
```

```python
# In Python
logging.info(f"[DEBUG] Compile image hidden states: shape={tensor.shape}, "
             f"range=[{tensor.min():.4f}, {tensor.max():.4f}]")
```

### 5. Order of Operations Matters

**Lesson**: Model transformations (reparameterization, quantization) must happen in the correct order.

**Correct Order**:
1. Create model architecture
2. Load pretrained weights
3. Reparameterize (fuse BatchNorm, etc.)
4. Run calibration forward pass
5. Apply quantization
6. Export to PTE

---

## Guide for Future VLM Integrations

### Step 1: Analyze the Model Architecture

1. **Identify components**:
   - Vision encoder architecture (ViT, ConvNet, hybrid)
   - Projection layer (MLP, linear)
   - Text decoder (LLaMA, Qwen, etc.)

2. **Check for problematic layers**:
   ```python
   for name, module in model.named_modules():
       if isinstance(module, nn.BatchNorm2d):
           print(f"{name}: running_var min={module.running_var.min():.2e}")
   ```

3. **Understand the tokenizer**:
   - Check `tokenizer_config.json` for `chat_template`
   - Look for default system prompts
   - Note special tokens (image, audio placeholders)

### Step 2: Create Configuration Files

1. **Decoder config** (`decoder/model_config.py`):
   - Inherit from appropriate base (Qwen, LLaMA, etc.)
   - Set correct model parameters

2. **Encoder config** (`encoder/encoder_config.py`):
   - Set image dimensions and patch count
   - Configure quantization recipe

3. **Special tokens** (`tokenizer.py`):
   ```python
   VLM_SPECIAL_TOKENS = {
       "your_model": {
           "image_token": "<image>",
           "image_placeholder": "<|vision_start|><|image_pad|><|vision_end|>",
       }
   }
   ```

### Step 3: Implement Vision Encoder Wrapper

```python
class YourVisionEncoder(nn.Module):
    def __init__(self, checkpoint_path: str = None, ...):
        super().__init__()
        self._build_model()
        if checkpoint_path:
            self._load_from_checkpoint(checkpoint_path)

    def _build_model(self):
        # Build vision tower
        # Build projection layer
        pass

    def _load_from_checkpoint(self, path):
        # Load weights
        # Reparameterize BatchNorm layers
        pass

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Return [batch, num_patches, hidden_dim]
        pass
```

### Step 4: Update C++ Runtime

1. **Add model version enum**:
   ```cpp
   enum class MultimodalDecoderModelVersion {
       // ...existing models...
       kYourModel,
   };
   ```

2. **Add version detection**:
   ```cpp
   if (model_version_str == "your_model") {
       return MultimodalDecoderModelVersion::kYourModel;
   }
   ```

3. **Add special tokens**:
   ```cpp
   case MultimodalDecoderModelVersion::kYourModel:
       return {"<image>", "<|endoftext|>"};
   ```

4. **Add prompt formatting**:
   ```cpp
   case MultimodalDecoderModelVersion::kYourModel:
       // Format prompt according to model's chat template
       break;
   ```

### Step 5: Test and Validate

1. **Compile and check output**:
   ```bash
   python llama.py --decoder_model your_model --compile_only ...
   ```

2. **Compare embeddings**:
   ```python
   compile_hidden = np.fromfile("debug_compile_image_hidden_states.raw", dtype=np.float32)
   runtime_hidden = np.fromfile("debug_runtime_image_hidden_states.raw", dtype=np.float32)
   correlation = np.corrcoef(compile_hidden, runtime_hidden)[0,1]
   print(f"Correlation: {correlation}")  # Should be > 0.99
   ```

3. **Run inference**:
   ```bash
   python llama.py --decoder_model your_model --pre_gen_pte your_model_hybrid ...
   ```

---

## Testing and Validation

### Quick Validation Script

```python
import numpy as np

def validate_embeddings(compile_path, runtime_path):
    compile_data = np.fromfile(compile_path, dtype=np.float32)
    runtime_data = np.fromfile(runtime_path, dtype=np.float32)

    if compile_data.shape != runtime_data.shape:
        print(f"FAIL: Shape mismatch {compile_data.shape} vs {runtime_data.shape}")
        return False

    diff = np.abs(compile_data - runtime_data)
    corr = np.corrcoef(compile_data, runtime_data)[0,1]

    print(f"Correlation: {corr:.6f}")
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")

    if corr < 0.99:
        print("FAIL: Low correlation - check quantization or reparameterization")
        return False

    print("PASS: Embeddings match")
    return True

# Usage
validate_embeddings(
    "debug_compile_image_hidden_states.raw",
    "debug_runtime_image_hidden_states.raw"
)
```

### Debugging Checklist

- [ ] Vision encoder input matches between compile/runtime
- [ ] Vision encoder output matches (correlation > 0.99)
- [ ] Text embeddings match (correlation > 0.99)
- [ ] Merged embeddings match (correlation > 0.99)
- [ ] Token count matches between compile/runtime
- [ ] Model generates coherent, relevant output

---

## Performance Considerations

### Vision Encoder Performance

| Configuration | PTE Size | Latency | Accuracy |
|---------------|----------|---------|----------|
| FP16 (no quant) | 266 MB | 32s | Best |
| 16a8w | 140 MB | ~20s | May degrade |
| 16a16w | ~180 MB | ~25s | Good (untested) |

### Recommendations

1. **Start with FP16** for correctness validation
2. **Try quantization** only after FP16 works correctly
3. **Use 16a16w** if 16a8w has accuracy issues
4. **Consider per-channel** quantization for linear layers

---

## Files Modified for FastVLM Integration

### New Files Created
| File | Description |
|------|-------------|
| `model/mci.py` | MCI (Multi-scale Convolutional Image) encoder components, ConvFFN with reparameterization |
| `model/fastvit.py` | FastViT/FastViTHD architecture implementation |
| `model/vision_encoder.py` | `FastVLMVisionEncoder` class wrapping the vision tower |
| `examples/models/fastvlm/` | Standalone FastVLM example with configs and weight conversion |

### Modified Files
| File | Changes |
|------|---------|
| `decoder/model_config.py` | Added `FastVLM_0_5B` config inheriting from Qwen |
| `decoder_constants.py` | Added FastVLM decoder model constant |
| `encoder/encoder_config.py` | Added `FastVLMEncoder` config with FP16 settings |
| `encoder/encoder_quant_recipe.py` | Added `FastVLM_Encoder_QuantRecipe` (16a8w) |
| `tokenizer.py` | Added FastVLM special tokens and VLM_SPECIAL_TOKENS entry |
| `wrappers.py` | Added FastVLM vision encoder integration in MultiModalManager |
| `decoder_utils.py` | Added debug logging for embeddings comparison |
| `qnn_multimodal_runner.cpp` | Added kFastvlm enum, special tokens, prompt formatting with default system prompt |
| `multimodal_runner.cpp` | Added debug output for vision encoder and merged embeddings |
| `encoder.cpp` | Added debug logging for vision encoder input/output |
| `static_llm_quant_recipe.py` | Added FastVLM quantization recipe for decoder |
| `dataset.py` | Updated calibration data handling for FastVLM |

### Key Code Changes

**1. ConvFFN Reparameterization** (`model/mci.py:~line 200`):
```python
def reparameterize(self) -> None:
    """Fuse Conv2d and BatchNorm2d into a single Conv2d."""
    # Critical for numerical stability with small variances
```

**2. Vision Encoder Weight Loading** (`model/vision_encoder.py:~line 518`):
```python
# CRITICAL: Reparameterize AFTER loading weights
self._reparameterize_convffn()
```

**3. Default System Prompt** (`qnn_multimodal_runner.cpp:~line 240`):
```cpp
// FastVLM uses Qwen2 tokenizer which injects default system prompt
formatted_prompt.append("You are a helpful assistant.");
```

---

## References

- [FastVLM Paper](https://arxiv.org/abs/...) - Model architecture
- [FastViT Repository](https://github.com/apple/ml-fastvit) - Vision encoder
- [Qwen2 Documentation](https://huggingface.co/Qwen/Qwen2-0.5B) - Text decoder
- [ExecuTorch QNN Backend](https://pytorch.org/executorch/stable/backends/qualcomm.html) - QNN integration

---

*Document created: January 2026*
*Last updated: January 17, 2026*

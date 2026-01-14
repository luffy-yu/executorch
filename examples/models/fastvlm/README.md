# FastVLM-0.5B for ExecuTorch Qualcomm Backend

This directory contains the implementation of FastVLM-0.5B for the ExecuTorch Qualcomm backend.

## Model Overview

FastVLM-0.5B is a Vision Language Model from Apple that features:
- **Vision Encoder**: FastViTHD (1024x1024 input, 3072-dim output)
- **Text Decoder**: Qwen2-based (896 hidden dim, 24 layers, 151,936 vocab)
- **Multimodal Fusion**: MLP projector (mlp2x_gelu: 3072 → 896)
- **Total Parameters**: ~800M (0.5B language model + vision encoder)

## Architecture

```
Input Image (1024x1024)
    ↓
FastViTHD Vision Encoder
    ↓
Vision Features (256 patches × 3072 dim)
    ↓
MLP Projector (3072 → 896)
    ↓
Text Decoder (Qwen2)
    ↓
Generated Text
```

## Files

- `__init__.py` - FastVLMModel class
- `convert_weights.py` - Weight conversion from HuggingFace to Meta format
- `config/0_5b_config.json` - Model configuration
- `README.md` - This file

## Weight Conversion

Convert FastVLM weights from HuggingFace format to ExecuTorch Meta format:

```bash
python examples/models/fastvlm/convert_weights.py \
    /path/to/FastVLM-0.5B \
    /path/to/output/fastvlm_0_5b.pth
```

This extracts only the text decoder weights. The vision encoder and projector are handled separately by the Qualcomm backend.

## Usage with Qualcomm Backend

### Test Command

```bash
python -m backends.qualcomm.tests.test_qnn_delegate \
    TestExampleMultimodalityScript.test_static_vlm \
    --model_name fastvlm_0_5b \
    -b build-android \
    --executorch_root . \
    -a . \
    -m SM8750 \
    -s ${SERIAL_NUM}
```

### Integration Points

The FastVLM implementation integrates with:
- **Model Registration**: `examples/qualcomm/oss_scripts/llama/__init__.py`
- **Vision Encoder**: `examples/qualcomm/oss_scripts/llama/model/vision_encoder.py`
- **Encoder Config**: `examples/qualcomm/oss_scripts/llama/encoder/encoder_config.py`
- **Quantization**: `examples/qualcomm/oss_scripts/llama/static_llm_quant_recipe.py`
- **Decoder Version**: `examples/qualcomm/oss_scripts/llama/decoder_constants.py`

## Current Status

### ✅ Implemented
- Model directory structure
- Weight conversion (text decoder only)
- Configuration files
- MLP projector (mlp2x_gelu)
- Quantization recipe (16a4w)
- Model registration
- Test integration

### ⚠️ Partial Implementation
- **Vision Encoder (FastViTHD)**: The class is implemented with MLP projector, but the FastViT backbone needs to be loaded from the checkpoint or implemented from scratch.

### 📝 TODO
1. Complete FastViTHD vision tower implementation:
   - Option A: Load from checkpoint using timm or custom implementation
   - Option B: Extract architecture from downloaded model
2. Test weight conversion
3. Test model compilation with QNN backend
4. Benchmark on-device performance

## Vision Encoder Implementation

The vision encoder (`FastVLMVisionEncoder`) has two main components:

### 1. MLP Projector (✅ Implemented)
```python
nn.Sequential(
    nn.Linear(3072, 896),  # vision_hidden_size → projector_hidden_size
    nn.GELU(),
    nn.Linear(896, 896),
)
```

### 2. FastViTHD Vision Tower (⚠️ Needs Implementation)
The vision tower architecture needs to be implemented to complete the vision encoder. You can:

**Option A**: Load from checkpoint
```python
checkpoint_path = "/path/to/FastVLM-0.5B/model.safetensors"
encoder = FastVLMVisionEncoder(config, checkpoint_path=checkpoint_path)
```

**Option B**: Implement FastViT from scratch
- Use the architecture from `/home/n10288/Documents/Code/FastVLM-0.5B/llava_qwen.py`
- Extract relevant classes: FastViT, MobileOneBlock, RepMixerBlock, AttentionBlock
- Adapt for QNN backend (remove dynamic shapes, optimize for mobile)

## Model Configuration

### Text Decoder (Qwen2-based)
```json
{
  "dim": 896,
  "hidden_dim": 4864,
  "n_heads": 14,
  "n_kv_heads": 2,
  "n_layers": 24,
  "vocab_size": 151936,
  "rope_theta": 1000000.0
}
```

### Vision Encoder (FastViTHD)
- Image size: 1024×1024
- Patch size: 64
- Number of patches: 256 (16×16)
- Output dim: 3072

### Quantization
- Text decoder: 16a4w (activation 16-bit, weight 4-bit)
- Per-block quantization for conv2d operations
- Follows Qwen2.5 quantization pattern

## Performance Estimates

Based on model size and architecture:
- **SM8650**: ~55 tokens/sec (estimated)
- **SM8750**: ~60 tokens/sec (estimated)
- **Model Size**: ~700 MB (encoder 150MB + embedding 150MB + decoder 400MB)

## References

- **HuggingFace**: https://huggingface.co/apple/FastVLM-0.5B
- **Paper**: FastVLM: Efficient Vision Encoding for Vision Language Models (CVPR 2025)
- **Base Model**: Qwen2-0.5B
- **Similar Implementation**: SmolVLM (`examples/models/smolvlm/`)

## Support

For issues or questions:
1. Check the implementation plan: `FASTVLM_IMPLEMENTATION_PLAN.md`
2. Review session summary: `FASTVLM_SESSION2_SUMMARY.md`
3. Compare with SmolVLM implementation: `examples/models/smolvlm/`

## License

BSD-style license (same as ExecuTorch)

import argparse
import json
import os
from typing import Dict, Optional, Tuple

import torch

from executorch.examples.models.smollm3.convert_weights import load_checkpoint
from torchtune.models.convert_weights import get_mapped_key


def dequantize_mlx_weight(
    weight: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int = 64
) -> torch.Tensor:
    """
    Dequantize MLX int8 weight tensor to float32.

    MLX uses affine quantization with per-group scales and biases:
        dequantized = weight * scales + biases

    Args:
        weight: int8 tensor of shape (out_features, in_features)
        scales: float16 tensor of shape (out_features, num_groups)
        biases: float16 tensor of shape (out_features, num_groups)
        group_size: size of each quantization group (default 64)

    Returns:
        float32 tensor of shape (out_features, in_features)
    """
    out_features, in_features = weight.shape
    num_groups = scales.shape[1]

    # Reshape weight to groups: (out_features, num_groups, group_size)
    weight_grouped = weight.view(out_features, num_groups, group_size).float()

    # Expand scales and biases to match: (out_features, num_groups, 1)
    scales_expanded = scales.unsqueeze(-1).float()
    biases_expanded = biases.unsqueeze(-1).float()

    # Dequantize: weight * scale + bias
    dequantized = weight_grouped * scales_expanded + biases_expanded

    # Reshape back to (out_features, in_features)
    return dequantized.view(out_features, in_features)


def dequantize_mlx_state_dict(
    state_dict: Dict[str, torch.Tensor],
    group_size: int = 64
) -> Dict[str, torch.Tensor]:
    """
    Dequantize all MLX quantized weights in a state dict.

    Finds all int8 weight tensors that have corresponding scales/biases
    and dequantizes them to float32.

    Args:
        state_dict: State dict potentially containing MLX quantized weights
        group_size: MLX quantization group size (default 64)

    Returns:
        State dict with dequantized float32 weights
    """
    dequantized_dict = {}
    processed_keys = set()

    for key, value in state_dict.items():
        # Skip if already processed (scales/biases are handled with their weight)
        if key in processed_keys:
            continue

        # Check if this is a quantized weight (int8 with corresponding scales/biases)
        if value.dtype == torch.int8:
            scales_key = key.replace('.weight', '.scales')
            biases_key = key.replace('.weight', '.biases')

            if scales_key in state_dict and biases_key in state_dict:
                # Dequantize this weight
                scales = state_dict[scales_key]
                biases = state_dict[biases_key]
                dequantized = dequantize_mlx_weight(value, scales, biases, group_size)
                dequantized_dict[key] = dequantized
                processed_keys.add(scales_key)
                processed_keys.add(biases_key)
            else:
                # Int8 without scales/biases - keep as is (shouldn't happen)
                dequantized_dict[key] = value.float()
        elif key.endswith('.scales') or key.endswith('.biases'):
            # Skip - these are handled with their corresponding weight
            continue
        else:
            # Non-quantized weight - convert to float32 if needed
            if value.dtype in (torch.float16, torch.bfloat16):
                dequantized_dict[key] = value.float()
            else:
                dequantized_dict[key] = value

    return dequantized_dict

# FastVLM uses Qwen2 text decoder with LLaVA-style multimodal architecture
# Map from HuggingFace FastVLM format (model.* for text decoder) to Meta format
# Note: Meta format uses .weight for norms, not .scale
_FASTVLM_TO_META = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.norm.weight": "norm.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
}

# MLX format (from mlx_vlm.convert) uses language_model.model.* prefix
_MLX_FASTVLM_TO_META = {
    "language_model.model.embed_tokens.weight": "tok_embeddings.weight",
    "language_model.model.norm.weight": "norm.weight",
    "language_model.model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "language_model.model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
    "language_model.model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "language_model.model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
    "language_model.model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "language_model.model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
    "language_model.model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "language_model.model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "language_model.model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    "language_model.model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "language_model.model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "language_model.model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
}


def fastvlm_tune_to_meta(
    state_dict: Dict[str, torch.Tensor],
    config_path: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from FastVLM's format to Meta's format. This function
    doesn't handle any sharding or splitting of state dicts. It follows the
    state_dict IN -> state_dict OUT pattern.

    Only extracts the text decoder weights, excluding vision encoder and projector.
    Supports both original HuggingFace format and MLX converted format.
    For MLX quantized models, dequantizes int8 weights to float32.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in FastVLM's format.
        config_path (Optional[str]): Path to config.json for reading quantization config.

    Returns:
        Dict[str, torch.Tensor]: State dict in Meta's format.
    """
    # Detect format: MLX uses language_model.* prefix
    is_mlx_format = any(k.startswith("language_model.") for k in state_dict.keys())

    # Check if MLX quantized (has int8 weights with scales/biases)
    has_quantized_weights = any(
        v.dtype == torch.int8 for v in state_dict.values()
    )

    if is_mlx_format and has_quantized_weights:
        # Read group_size from config if available
        group_size = 64  # default
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
                quant_config = config.get("quantization", {})
                group_size = quant_config.get("group_size", 64)
                print(f"MLX quantization config: bits={quant_config.get('bits', 8)}, group_size={group_size}")

        # Dequantize MLX int8 weights to float32
        print("Dequantizing MLX int8 weights to float32...")
        state_dict = dequantize_mlx_state_dict(state_dict, group_size)

    mapping = _MLX_FASTVLM_TO_META if is_mlx_format else _FASTVLM_TO_META

    converted_text_model_state_dict = {}
    for key, value in state_dict.items():
        # Skip MLX quantization parameters (scales, biases) - should be handled by dequantize
        if key.endswith('.scales') or key.endswith('.biases'):
            continue
        try:
            new_key = get_mapped_key(key, mapping)
            # Ensure float32 for QNN backend
            if value.dtype in (torch.float16, torch.bfloat16):
                value = value.float()
            converted_text_model_state_dict[new_key] = value
        except:
            # only preserve parameters of text decoder
            pass

    # Add output layer (lm_head)
    # Handle tied embeddings: if lm_head.weight doesn't exist, use embed_tokens.weight
    if "lm_head.weight" in state_dict:
        output_weight = state_dict["lm_head.weight"]
    elif "language_model.model.embed_tokens.weight" in state_dict:
        # MLX format with tied embeddings
        output_weight = state_dict["language_model.model.embed_tokens.weight"]
    elif "model.embed_tokens.weight" in state_dict:
        # Original format with tied embeddings
        output_weight = state_dict["model.embed_tokens.weight"]
    else:
        output_weight = None

    if output_weight is not None:
        if output_weight.dtype in (torch.float16, torch.bfloat16):
            output_weight = output_weight.float()
        converted_text_model_state_dict["output.weight"] = output_weight

    return converted_text_model_state_dict


def convert_weights(input_dir: str, output_file: str) -> None:
    print("Loading checkpoint...")
    sd = load_checkpoint(input_dir)
    print("Converting checkpoint...")
    converted_sd = fastvlm_tune_to_meta(sd)
    print("Saving checkpoint...")
    torch.save(converted_sd, output_file)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FastVLM weights to Meta format."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to directory containing checkpoint files",
    )
    parser.add_argument("output", type=str, help="Path to the output checkpoint")

    args = parser.parse_args()
    convert_weights(args.input_dir, args.output)


if __name__ == "__main__":
    main()

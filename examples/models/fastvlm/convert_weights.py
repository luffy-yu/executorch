import argparse
from typing import Dict

import torch

from executorch.examples.models.smollm3.convert_weights import load_checkpoint
from torchtune.models.convert_weights import get_mapped_key

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


def fastvlm_tune_to_meta(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from FastVLM's format to Meta's format. This function
    doesn't handle any sharding or splitting of state dicts. It follows the
    state_dict IN -> state_dict OUT pattern.

    Only extracts the text decoder weights, excluding vision encoder and projector.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in FastVLM's format.

    Returns:
        Dict[str, torch.Tensor]: State dict in Meta's format.
    """
    converted_text_model_state_dict = {}
    for key, value in state_dict.items():
        try:
            new_key = get_mapped_key(key, _FASTVLM_TO_META)
            converted_text_model_state_dict[new_key] = value
        except:
            # only preserve parameters of text decoder
            pass

    # Add output layer (lm_head)
    converted_text_model_state_dict["output.weight"] = state_dict["lm_head.weight"]

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

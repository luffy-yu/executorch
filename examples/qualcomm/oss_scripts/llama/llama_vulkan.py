#!/usr/bin/env python3
# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export FastVLM model to ExecuTorch Vulkan backend.

This script exports the FastVLM vision encoder, text embedding, and text decoder
to Vulkan .pte files for GPU execution on Android devices.

No QNN dependency - runs entirely on GPU via Vulkan API.

Usage:
    python llama_vulkan.py \
        --checkpoint ~/models/fastvlm-0.5b \
        --output_dir fastvlm_vulkan

    # Or just export vision encoder
    python llama_vulkan.py \
        --checkpoint ~/models/fastvlm-0.5b \
        --output_dir fastvlm_vulkan \
        --vision_only
"""

import argparse
import logging
import os
import sys

import torch
from torch.export import export

# Set up logging
FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def export_to_vulkan(model, sample_inputs, output_path, model_name, dynamic_shapes=None, storage_type=None, op_blocklist=None):
    """
    Export a PyTorch model to Vulkan .pte file.

    Args:
        model: PyTorch model to export
        sample_inputs: Tuple of example inputs for tracing
        output_path: Directory to save the .pte file
        model_name: Name for the output file (without .pte extension)
        dynamic_shapes: Optional dynamic shape specifications
        storage_type: Optional VkStorageType override (BUFFER, TEXTURE_2D, TEXTURE_3D)
        op_blocklist: Optional list of EdgeOpOverload objects to blocklist from Vulkan

    Returns:
        Path to the exported .pte file
    """
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    logger.info(f"Exporting {model_name} to Vulkan backend...")

    # Export to ATen dialect
    logger.info("  Step 1: Exporting to ATen dialect...")
    exported_program = export(
        model,
        sample_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=True,
    )

    # Lower to Vulkan
    logger.info("  Step 2: Lowering to Vulkan backend...")
    vulkan_options = {
        "require_dynamic_shapes": dynamic_shapes is not None,
        "skip_bool_tensors": True,  # Required for Vulkan backend
    }
    if storage_type is not None:
        from executorch.backends.vulkan.serialization.vulkan_graph_schema import VkStorageType
        vulkan_options["storage_type_override"] = VkStorageType[storage_type.upper()]
        logger.info(f"  Using storage type override: {storage_type}")

    partitioner_kwargs = {}
    if op_blocklist is not None:
        partitioner_kwargs["operator_blocklist"] = op_blocklist
        logger.info(f"  Blocklisting {len(op_blocklist)} ops from Vulkan delegation")

    edge_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[VulkanPartitioner(vulkan_options, **partitioner_kwargs)],
        compile_config=EdgeCompileConfig(_skip_dim_order=False, _check_ir_validity=False),
    )

    # Convert to ExecutorchProgram
    logger.info("  Step 3: Converting to ExecutorchProgram...")
    et_program = edge_program.to_executorch()

    # Save .pte file
    pte_path = os.path.join(output_path, f"{model_name}.pte")
    with open(pte_path, "wb") as f:
        f.write(et_program.buffer)

    file_size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    logger.info(f"  Exported {model_name} to {pte_path} ({file_size_mb:.2f} MB)")

    return pte_path


def export_vision_encoder(checkpoint_path, output_dir, storage_type=None, op_blocklist=None, output_name="vision_encoder"):
    """
    Export FastVLM vision encoder to Vulkan.

    The vision encoder uses FastViTHD architecture with:
    - Input: (1, 3, 1024, 1024) image tensor
    - Output: (1, 256, 896) vision features (256 patches, 896 dim)
    """
    from executorch.examples.qualcomm.oss_scripts.llama.model.vision_encoder import (
        FastVLMVisionEncoder,
    )
    from transformers import AutoConfig

    logger.info("=" * 60)
    logger.info("Exporting FastVLM Vision Encoder")
    logger.info("=" * 60)

    # Load config from checkpoint
    logger.info(f"Loading config from {checkpoint_path}")
    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)

    # Create vision encoder
    logger.info("Creating FastVLMVisionEncoder...")
    vision_encoder = FastVLMVisionEncoder(
        config,
        img_resized_h=1024,
        img_resized_w=1024,
        checkpoint_path=checkpoint_path,
    )
    vision_encoder.eval()

    # Example input: (batch=1, channels=3, height=1024, width=1024)
    sample_inputs = (torch.randn(1, 3, 1024, 1024),)

    # Test forward pass
    logger.info("Testing forward pass...")
    with torch.no_grad():
        output = vision_encoder(*sample_inputs)
    logger.info(f"  Vision encoder output shape: {output.shape}")
    logger.info(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Export to Vulkan
    pte_path = export_to_vulkan(
        vision_encoder,
        sample_inputs,
        output_dir,
        output_name,
        storage_type=storage_type,
        op_blocklist=op_blocklist,
    )

    return pte_path


def export_text_embedding(checkpoint_path, output_dir, vocab_size=151936, hidden_size=896):
    """
    Export text embedding layer to Vulkan.

    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory for .pte file
        vocab_size: Vocabulary size (default: 151936 for Qwen2)
        hidden_size: Hidden dimension (default: 896 for Qwen2-0.5B)
    """
    logger.info("=" * 60)
    logger.info("Exporting Text Embedding Layer")
    logger.info("=" * 60)

    # Create a simple embedding wrapper
    class TextEmbedding(torch.nn.Module):
        def __init__(self, vocab_size, hidden_size):
            super().__init__()
            # Use float32 for Vulkan backend compatibility
            self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
            self.embed_tokens.weight.data = self.embed_tokens.weight.data.to(torch.float32)

        def forward(self, input_ids):
            return self.embed_tokens(input_ids)

    # Create and load weights from checkpoint
    logger.info(f"Creating embedding layer (vocab={vocab_size}, hidden={hidden_size})")
    embedding = TextEmbedding(vocab_size, hidden_size)

    # Try to load weights from checkpoint
    try:
        from safetensors.torch import load_file

        safetensor_path = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(safetensor_path):
            logger.info(f"Loading embedding weights from {safetensor_path}")
            state_dict = load_file(safetensor_path)
            # Find embedding weights
            for key in state_dict:
                if "embed_tokens" in key and "weight" in key:
                    weight = state_dict[key]
                    # Vulkan requires fp32 - convert from bfloat16 or float16
                    if weight.dtype in (torch.bfloat16, torch.float16):
                        weight = weight.to(torch.float32)
                    embedding.embed_tokens.weight.data = weight
                    logger.info(f"  Loaded embedding weights from {key}")
                    break
    except Exception as e:
        logger.warning(f"Could not load embedding weights: {e}")
        logger.warning("Using random initialization")

    embedding.eval()

    # Example input: token IDs
    sample_inputs = (torch.randint(0, vocab_size, (1, 16), dtype=torch.long),)

    # Test forward pass
    logger.info("Testing forward pass...")
    with torch.no_grad():
        output = embedding(*sample_inputs)
    logger.info(f"  Embedding output shape: {output.shape}")

    # Export to Vulkan
    pte_path = export_to_vulkan(
        embedding,
        sample_inputs,
        output_dir,
        "text_embedding",
    )

    return pte_path


def export_text_decoder(checkpoint_path, output_dir, max_seq_len=1024):
    """
    Export text decoder (Qwen2) to Vulkan.

    Note: This is a simplified export. For full KV-cache support,
    use the standard Llama export script with -V flag:

        python -m executorch.examples.models.llama.export_llama \
            --checkpoint {checkpoint_path} \
            -V --vulkan-force-fp16 \
            --use_kv_cache
    """
    logger.info("=" * 60)
    logger.info("Exporting Text Decoder (Qwen2)")
    logger.info("=" * 60)

    logger.info("NOTE: For full text decoder export with KV cache support,")
    logger.info("use the standard Llama export script with Vulkan flag:")
    logger.info("")
    logger.info("  python -m executorch.examples.models.llama.export_llama \\")
    logger.info(f"      --checkpoint {checkpoint_path} \\")
    logger.info("      -V --vulkan-force-fp16 \\")
    logger.info("      --use_kv_cache")
    logger.info("")

    # For now, skip decoder export - direct user to existing script
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Export FastVLM model to ExecuTorch Vulkan backend"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to FastVLM checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        default="fastvlm_vulkan",
        help="Output directory for .pte files",
    )
    parser.add_argument(
        "--vision_only",
        action="store_true",
        help="Only export vision encoder",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=151936,
        help="Vocabulary size for text embedding",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=896,
        help="Hidden dimension size",
    )
    parser.add_argument(
        "--storage_type",
        type=str,
        default=None,
        choices=["buffer", "texture_2d", "texture_3d"],
        help="Vulkan storage type override (default: texture_3d). "
        "Use 'buffer' or 'texture_2d' to work around Adreno shader compiler crashes.",
    )
    parser.add_argument(
        "--op_blocklist",
        type=str,
        default=None,
        help="Comma-separated list of ops to blocklist from Vulkan delegation. "
        "Blocklisted ops fall back to CPU. "
        "Format: aten.mean.dim,aten.softmax.int "
        "Use this to work around Adreno shader compiler crashes for specific ops.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="vision_encoder",
        help="Name for the output .pte file (without extension)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Parse op blocklist
    op_blocklist = None
    if args.op_blocklist:
        from executorch.exir.dialects._ops import ops as exir_ops
        op_blocklist = []
        for name in args.op_blocklist.split(","):
            name = name.strip()
            # Format: aten.op_name.overload
            parts = name.replace("aten.", "").split(".")
            try:
                edge_op = getattr(getattr(exir_ops.edge.aten, parts[0]), parts[1] if len(parts) > 1 else "default")
                op_blocklist.append(edge_op)
                logger.info(f"  Blocklisting: {name} -> {edge_op}")
            except AttributeError:
                logger.warning(f"  Could not resolve edge op: {name}")
        logger.info(f"Total ops blocklisted: {len(op_blocklist)}")

    exported_files = []

    # Export vision encoder
    try:
        pte_path = export_vision_encoder(
            args.checkpoint, args.output_dir,
            storage_type=args.storage_type,
            op_blocklist=op_blocklist,
            output_name=args.output_name,
        )
        exported_files.append(pte_path)
    except Exception as e:
        logger.error(f"Failed to export vision encoder: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    if not args.vision_only:
        # Export text embedding
        try:
            pte_path = export_text_embedding(
                args.checkpoint,
                args.output_dir,
                vocab_size=args.vocab_size,
                hidden_size=args.hidden_size,
            )
            if pte_path:
                exported_files.append(pte_path)
        except Exception as e:
            logger.error(f"Failed to export text embedding: {e}")
            import traceback

            traceback.print_exc()

        # Export text decoder (prints instructions)
        export_text_decoder(args.checkpoint, args.output_dir)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Export Summary")
    logger.info("=" * 60)
    logger.info(f"Exported {len(exported_files)} file(s):")
    for f in exported_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        logger.info(f"  - {f} ({size_mb:.2f} MB)")

    logger.info("")
    logger.info("To run on Android device:")
    logger.info(f"  adb push {args.output_dir}/*.pte /data/local/tmp/fastvlm/")
    logger.info(
        "  adb shell '/data/local/tmp/vulkan_executor_runner --model_path /data/local/tmp/fastvlm/vision_encoder.pte'"
    )


if __name__ == "__main__":
    main()

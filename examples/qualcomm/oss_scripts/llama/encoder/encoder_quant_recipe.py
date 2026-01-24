# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from executorch.backends.qualcomm.quantizer.quant_recipe import (
    QuantGranularity,
    QuantRecipe,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from torchao.quantization.pt2e import HistogramObserver, MinMaxObserver


class EncoderQuantRecipe:
    """
    Qualcomm's Encoder quantization recipe.
    """

    def __init__(self):
        self.recipe: Optional[QuantRecipe] = None

        self.default_quant_dtype = getattr(self, "default_quant_dtype", None)
        if self.default_quant_dtype is None:
            raise ValueError("default_quant_dtype must be defined in the recipe.")

    def annotate(self, graph_module: torch.fx.GraphModule):
        self.recipe.annotate(graph_module)


class InternVL3_Encoder_QuantRecipe(EncoderQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a8w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = QuantRecipe(
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_TENSOR,
            verbose=verbose,
        ).add_node_target(
            {
                torch.ops.aten.linear.default,
            },
            QuantDtype.use_16a8w,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_CHANNEL,
        )


class SmolVLM_Encoder_QuantRecipe(EncoderQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a8w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = QuantRecipe(
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_TENSOR,
            verbose=verbose,
        ).add_node_target(
            {
                torch.ops.aten.linear.default,
            },
            QuantDtype.use_16a8w,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_CHANNEL,
        )


class FastVLM_Encoder_QuantRecipe(EncoderQuantRecipe):
    """
    Quantization recipe for FastVLM vision encoder (FastViTHD).

    FastViTHD uses primarily Conv2d operations (MobileOneBlock, ReparamLargeKernelConv,
    ConvFFN, SEBlock) with few linear layers (only in MHSA for attention blocks).

    Uses 16a16w (16-bit activations, 16-bit weights) for maximum precision.
    This essentially keeps FP16 precision but goes through the quantization
    infrastructure, which may help with QNN backend compatibility.

    NOTE: 16a8w was tried but caused hallucination issues at QNN runtime.
    Using 16a16w as a baseline to verify if the issue is with weight quantization.
    """
    default_quant_dtype = QuantDtype.use_16a16w

    def __init__(self, verbose: bool = False):
        super().__init__()

        # Use 16a16w for FastViTHD - full precision through quantization infra
        self.recipe = QuantRecipe(
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_TENSOR,
            verbose=verbose,
        ).add_node_target(
            {
                # PyTorch may export Conv2d as either conv2d.default or convolution.default
                torch.ops.aten.conv2d.default,
                torch.ops.aten.convolution.default,
            },
            QuantDtype.use_16a16w,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_CHANNEL,
        ).add_node_target(
            {
                torch.ops.aten.linear.default,
            },
            QuantDtype.use_16a16w,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_CHANNEL,
        )

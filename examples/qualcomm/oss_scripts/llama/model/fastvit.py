# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastViT implementation extracted from FastVLM model.
Adapted for ExecuTorch QNN backend with static shapes and inference mode.

Original source: https://huggingface.co/apple/FastVLM-0.5B
Based on: FastViT: A Fast Hybrid Vision Transformer
"""

from functools import partial
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SEBlock(nn.Module):
    """Squeeze and Excite module for channel attention."""

    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * rd_ratio),
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.expand = nn.Conv2d(
            in_channels=int(in_channels * rd_ratio),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        b, c, h, w = inputs.size()
        # Fixed pooling size for QNN backend
        x = F.avg_pool2d(inputs, kernel_size=[16, 16])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class MobileOneBlock(nn.Module):
    """
    MobileOne building block for inference.
    Simplified for QNN backend - inference mode only.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        inference_mode: bool = True,  # Always True for QNN
        use_se: bool = False,
        use_act: bool = True,
        use_scale_branch: bool = True,
        num_conv_branches: int = 1,
        activation: nn.Module = None,
    ) -> None:
        super(MobileOneBlock, self).__init__()
        self.inference_mode = True  # Force inference mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        if activation is None:
            activation = nn.GELU()

        # Squeeze-Excite
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()

        # Activation
        if use_act:
            self.activation = activation
        else:
            self.activation = nn.Identity()

        # For inference mode, use single reparameterized convolution
        self.reparam_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - inference mode only."""
        x = self.reparam_conv(x)
        x = self.se(x)
        x = self.activation(x)
        return x


class LayerNormChannel(nn.Module):
    """Layer normalization for channels-first tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-05) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class RepCPE(nn.Module):
    """Reparameterized Convolutional Position Encoding."""

    def __init__(self, in_channels: int, spatial_shape: Tuple[int, int] = (7, 7)):
        super(RepCPE, self).__init__()
        self.spatial_shape = spatial_shape
        self.pos_embed = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=spatial_shape,
            padding=int(spatial_shape[0] // 2),
            groups=in_channels,
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_embed(x)


class RepMixerBlock(nn.Module):
    """RepMixer block with token mixing and channel mixing."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = True,
    ):
        super().__init__()
        self.token_mixer = MobileOneBlock(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=dim,
            use_act=False,
            use_scale_branch=False,
            num_conv_branches=1,
            inference_mode=True,
        )

        # Channel mixing (MLP)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            MobileOneBlock(
                in_channels=dim,
                out_channels=mlp_hidden_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                inference_mode=True,
            ),
            MobileOneBlock(
                in_channels=mlp_hidden_dim,
                out_channels=dim,
                kernel_size=1,
                stride=1,
                padding=0,
                use_act=False,
                inference_mode=True,
            ),
        )

        # Layer scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim, 1, 1)
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim, 1, 1)
            )
        else:
            self.layer_scale_1 = None
            self.layer_scale_2 = None

    def forward(self, x: Tensor) -> Tensor:
        # Token mixing
        if self.layer_scale_1 is not None:
            x = x + self.layer_scale_1 * self.token_mixer(x)
        else:
            x = x + self.token_mixer(x)

        # Channel mixing
        if self.layer_scale_2 is not None:
            x = x + self.layer_scale_2 * self.mlp(x)
        else:
            x = x + self.mlp(x)

        return x


class AttentionBlock(nn.Module):
    """Multi-head self-attention block."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        num_heads: int = 8,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        # MLP
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1),
        )

        # Layer scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim, 1, 1)
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim, 1, 1)
            )
        else:
            self.layer_scale_1 = None
            self.layer_scale_2 = None

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        # Attention
        x_norm = self.norm1(x)
        x_flat = x_norm.flatten(2).transpose(1, 2)  # (B, H*W, C)

        qkv = self.qkv(x_flat).reshape(B, H * W, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        if self.layer_scale_1 is not None:
            x = x + self.layer_scale_1 * out
        else:
            x = x + out

        # MLP
        x_mlp = self.norm2(x)
        x_mlp = self.mlp(x_mlp)

        if self.layer_scale_2 is not None:
            x = x + self.layer_scale_2 * x_mlp
        else:
            x = x + x_mlp

        return x


class GlobalPool2D(nn.Module):
    """Global pooling with linear projection."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        scale = in_dim**-0.5
        self.proj = nn.Parameter(scale * torch.randn(size=(in_dim, out_dim)))
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 4, f"Expected 4D input, got {x.dim()}D"
        # Global average pooling
        x = torch.mean(x, dim=[-2, -1], keepdim=False)
        # Linear projection
        x = x @ self.proj
        return x


def basic_blocks(
    dim: int,
    block_index: int,
    num_blocks: List[int],
    token_mixer_type: str,
    kernel_size: int = 3,
    mlp_ratio: float = 4.0,
    act_layer: nn.Module = nn.GELU,
    norm_layer: nn.Module = nn.BatchNorm2d,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    use_layer_scale: bool = True,
    layer_scale_init_value: float = 1e-5,
    inference_mode: bool = True,
) -> nn.Sequential:
    """Build FastViT blocks within a stage."""
    blocks = []
    for block_idx in range(num_blocks[block_index]):
        block_dpr = (
            drop_path_rate
            * (block_idx + sum(num_blocks[:block_index]))
            / (sum(num_blocks) - 1)
        )
        if token_mixer_type == "repmixer":
            blocks.append(
                RepMixerBlock(
                    dim,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    inference_mode=True,
                )
            )
        elif token_mixer_type == "attention":
            blocks.append(
                AttentionBlock(
                    dim,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                )
            )
        else:
            raise ValueError(f"Token mixer type: {token_mixer_type} not supported")

    return nn.Sequential(*blocks)


def convolutional_stem(
    in_channels: int,
    out_channels: int,
    inference_mode: bool = False,
    use_scale_branch: bool = True,
) -> nn.Sequential:
    """
    Build convolutional stem with MobileOne blocks.

    Creates a 3-layer stem that downsamples the input by 4x (2 stride-2 convs + 1 stride-1 conv).

    Args:
        in_channels: Number of input channels (usually 3 for RGB).
        out_channels: Number of output channels.
        inference_mode: Flag to instantiate model in inference mode.
        use_scale_branch: Whether to use scale branch in MobileOneBlock.

    Returns:
        nn.Sequential object with 3 MobileOneBlock layers.
    """
    return nn.Sequential(
        MobileOneBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
            use_scale_branch=use_scale_branch,
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=out_channels,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
            use_scale_branch=use_scale_branch,
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
            use_scale_branch=use_scale_branch,
        ),
    )


class FastViT(nn.Module):
    """
    FastViT architecture for vision encoding.
    Adapted for QNN backend - inference mode only, static shapes.
    """

    def __init__(
        self,
        layers: List[int],
        token_mixers: Tuple[str, ...],
        embed_dims: List[int] = None,
        mlp_ratios: List[float] = None,
        downsamples: List[bool] = None,
        repmixer_kernel_size: int = 3,
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.GELU,
        num_classes: int = 1000,
        pos_embs: List = None,
        down_patch_size: int = 7,
        down_stride: int = 2,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        cls_ratio: float = 2.0,
        inference_mode: bool = True,
        stem_scale_branch: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        # Patch embedding stem - 3-layer convolutional stem
        self.patch_embed = convolutional_stem(
            in_channels=3,
            out_channels=embed_dims[0],
            inference_mode=True,
            use_scale_branch=stem_scale_branch,
        )

        # Build network stages
        network = []
        for i in range(len(layers)):
            # Positional encoding
            if pos_embs[i] is not None:
                network.append(pos_embs[i](embed_dims[i]))

            # Blocks
            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                token_mixers[i],
                kernel_size=repmixer_kernel_size,
                mlp_ratio=mlp_ratios[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                inference_mode=True,
            )
            network.append(stage)

            # Downsampling
            if i < len(layers) - 1 and downsamples[i]:
                network.append(
                    MobileOneBlock(
                        in_channels=embed_dims[i],
                        out_channels=embed_dims[i + 1],
                        kernel_size=down_patch_size,
                        stride=down_stride,
                        padding=down_patch_size // 2,
                        inference_mode=True,
                        use_se=False,
                    )
                )

        self.network = nn.ModuleList(network)

        # Classification head
        self.conv_exp = MobileOneBlock(
            in_channels=embed_dims[-1],
            out_channels=int(embed_dims[-1] * cls_ratio),
            kernel_size=3,
            stride=1,
            padding=1,
            groups=embed_dims[-1],
            inference_mode=True,
            use_se=True,
            num_conv_branches=1,
        )

        self.head = (
            nn.Linear(int(embed_dims[-1] * cls_ratio), num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def forward_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Input embedding."""
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Process through network."""
        for block in self.network:
            x = block(x)
        return x

    def forward(
        self, x: torch.Tensor, return_image_embeddings: bool = False
    ) -> dict:
        """Forward pass."""
        # Input embedding
        x = self.forward_embeddings(x)

        # Through backbone
        x = self.forward_tokens(x)

        # Classification head
        x = self.conv_exp(x)

        if return_image_embeddings:
            # Return both logits and embeddings
            cls_out = self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))
            return {
                "logits": cls_out,
                "image_embeddings": x,  # (B, C, H, W)
            }
        else:
            # Classification only
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
            return self.head(x)


def fastvithd(pretrained: bool = False, **kwargs):
    """
    Instantiate FastViTHD model variant for FastVLM.

    Architecture:
    - 5 stages with [2, 12, 24, 4, 2] blocks
    - Embedding dims: [96, 192, 384, 768, 1536]
    - Token mixers: repmixer for first 3 stages, attention for last 2
    """
    layers = [2, 12, 24, 4, 2]
    embed_dims = [96, 192, 384, 768, 1536]
    mlp_ratios = [4, 4, 4, 4, 4]
    downsamples = [True, True, True, True, True]
    pos_embs = [
        None,
        None,
        None,
        partial(RepCPE, spatial_shape=(7, 7)),
        partial(RepCPE, spatial_shape=(7, 7)),
    ]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention", "attention")

    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        norm_layer=LayerNormChannel,
        stem_scale_branch=False,
        inference_mode=True,
        **kwargs,
    )

    if pretrained:
        raise ValueError("Pretrained loading not implemented - load weights manually")

    return model

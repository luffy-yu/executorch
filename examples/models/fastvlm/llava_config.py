#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
Minimal LlavaConfig registration for FastVLM support.

This module registers the 'llava_qwen2' model type with transformers AutoConfig,
allowing AutoConfig.from_pretrained() to work with FastVLM checkpoints.

The full model implementation is not needed here since executorch uses its own
Static LLaMA architecture for inference. We only need the config class to be
registered so that the checkpoint config.json can be loaded.

Original source: https://github.com/apple/ml-fastvlm
"""

from transformers import AutoConfig, Qwen2Config


class LlavaConfig(Qwen2Config):
    """
    Configuration class for LLaVA-Qwen2 models (FastVLM).

    This extends Qwen2Config with the 'llava_qwen2' model type identifier,
    enabling transformers to recognize FastVLM checkpoint configurations.
    """

    model_type = "llava_qwen2"


# Register the config with transformers AutoConfig
# This allows AutoConfig.from_pretrained() to work with FastVLM checkpoints
AutoConfig.register("llava_qwen2", LlavaConfig)

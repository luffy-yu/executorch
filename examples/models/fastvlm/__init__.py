# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.examples.models.llama.model import Llama2Model
from executorch.examples.models.fastvlm.convert_weights import convert_weights

# Import llava_config to register the 'llava_qwen2' model type with transformers AutoConfig
# This is required for AutoConfig.from_pretrained() to work with FastVLM checkpoints
from executorch.examples.models.fastvlm import llava_config  # noqa: F401


class FastVLMModel(Llama2Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


__all__ = [
    "FastVLMModel",
    "convert_weights",
]

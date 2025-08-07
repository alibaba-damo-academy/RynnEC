# Adopted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/configuration_siglip.py.
# Below is the original copyright:
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""VideoLLaMA3 vision encoder model configuration."""
import os
from typing import Union

from transformers import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

class Videollama3VisionEncoderConfig(PretrainedConfig):

    model_type = "videollama3_vision_encoder"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
    #     cls._set_token_in_kwargs(kwargs)

    #     config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

    #     p
    #     config_dict = config_dict["vision_config"]

    #     if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
    #         logger.warning(
    #             f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
    #             f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
    #         )

    #     return cls.from_dict(config_dict, **kwargs)

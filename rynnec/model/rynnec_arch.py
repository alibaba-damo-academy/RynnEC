# Adopted from: https://github.com/DAMO-NLP-SG/VideoLLaMA3. 
# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
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

import os
import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import einops
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np

from ..constants import IGNORE_INDEX, MODAL_INDEX_MAP, NUM_FRAMES
from .encoder import build_vision_encoder
from .projector import build_vision_projector, load_mm_projector
from .region_encoder import build_region_encoder
from ..mm_utils import reshape_images_to_raw_grid


def spatial_downsampling(features, grid_thws, stride=2):
    n, c = features.shape

    flatten_grid_thws = torch.cat([grid_thw for batch_grid_thws in grid_thws for grid_thw in batch_grid_thws])
    split_sizes = [grid_thw.prod() for grid_thw in flatten_grid_thws]
    features = torch.split(features, split_sizes)

    new_features = []
    for feature, grid_thw in zip(features, flatten_grid_thws):
        # NOTE: adapted for reshape in image processor 
        feature = feature.view(grid_thw[0], grid_thw[1] // stride, grid_thw[2] // stride, stride, stride,  c).permute(0, 1, 3, 2, 4, 5)
        feature = feature.reshape(grid_thw[0], grid_thw[1], grid_thw[2], c).permute(0, 3, 1, 2)
        # NOTE: previous version model is align_corners=True
        new_feature = torch.nn.functional.interpolate(feature, (math.ceil(grid_thw[1] / stride), math.ceil(grid_thw[2] / stride)), mode='bilinear')
        # new_feature = nn.functional.avg_pool2d(feature, stride)
        # new_feature = nn.functional.max_pool2d(feature, stride)
        new_features.append(new_feature.permute(0, 2, 3, 1).view(-1, c))
    new_features = torch.cat(new_features)

    return new_features


class RynnecMetaModel:

    def __init__(self, config):
        super(RynnecMetaModel, self).__init__(config)

        if hasattr(config, "vision_encoder") or hasattr(config, "mm_vision_encoder"):
            self.vision_encoder = build_vision_encoder(config, delay_load=False)
            self.mm_projector = build_vision_projector(config, config.mm_hidden_size)
        self.region_encoder = build_region_encoder(config, config.mm_hidden_size) 

    def get_vision_encoder(self):
        vision_encoder = getattr(self, 'vision_encoder', None)
        if type(vision_encoder) is list:
            vision_encoder = vision_encoder[0]
        return vision_encoder

    def get_mm_projector(self):
        return self.mm_projector

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_encoder = model_args.vision_encoder
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_projector = model_args.pretrain_mm_projector

        self.config.mm_vision_encoder = vision_encoder

        if self.get_vision_encoder() is None:
            vision_encoder = build_vision_encoder(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_encoder = [vision_encoder]
            else:
                self.vision_encoder = vision_encoder
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_encoder = self.vision_encoder[0]
            else:
                vision_encoder = self.vision_encoder
            # NOTE: only compatible with delay_load encoder
            # vision_encoder.load_model(vision_encoder.cfg_only)

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_encoder.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_projector is not None:
            if os.path.exists(pretrain_mm_projector):
                is_local = True
                if os.path.isdir(pretrain_mm_projector):
                    mm_projector_weights = load_mm_projector(pretrain_mm_projector)
                else:
                    mm_projector_weights = torch.load(pretrain_mm_projector, map_location='cpu')
            else:
                # Support loading projector weights from remote HuggingFace model hub
                is_local = False
                pretrain_mm_projector = pretrain_mm_projector.replace('mm_projector.bin', '')
                pretrain_mm_projector = pretrain_mm_projector.strip('/').strip('\\').strip()
                mm_projector_weights = load_mm_projector(pretrain_mm_projector)

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)


class RynnecMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def num_frames(self):
        if hasattr(self.config, 'num_frames'):
            return self.config.num_frames
        else:
            return NUM_FRAMES

    def spatial_merge_size(self):
        if hasattr(self.config, 'spatial_merge_size'):
            return self.config.spatial_merge_size
        else:
            return 1
    
    def get_vision_encoder(self):
        return self.get_model().get_vision_encoder()

    def get_mm_projector(self):
        return self.get_model().get_mm_projector()

    def encode_images(
        self,
        pixel_values: torch.FloatTensor,
        grid_sizes: torch.LongTensor,
        merge_sizes: torch.LongTensor,
    ):
        mm_features, mm_features_raw = self.get_model().get_vision_encoder()(
            pixel_values=pixel_values,
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
        )
        mm_features = self.get_model().mm_projector(mm_features)
        return mm_features, mm_features_raw

    def _get_valid_visual_tokens(
        self,
        mm_features: torch.FloatTensor,
        batched_num_patches: torch.LongTensor,
        modals: List[str],
    ):
        valid_masks = []
        for num_patches, modal in zip(batched_num_patches, modals):
            valid_mask = torch.full((num_patches, ), modal != "text", dtype=torch.bool, device=mm_features.device)
            valid_masks.append(valid_mask)
        mm_features = mm_features[torch.cat(valid_masks)]
        return mm_features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        masks=None, 
        mask_ids = None
    ):
        vision_encoder = self.get_vision_encoder()
        # NOTE: text-only situation
        if vision_encoder is None or pixel_values is None or input_ids.shape[1] == 1:
            return input_ids, attention_mask, position_ids, past_key_values, None, labels

        # 1. flatten text inputs
        B, N = input_ids.shape
        input_ids = input_ids.view(B * N)
        if attention_mask is not None:
            attention_mask = attention_mask.view(B * N)
        if position_ids is not None:
            position_ids = position_ids.view(B * N)
        if labels is not None:
            labels = labels.view(B * N)

        # 2. embed visual tokens
        batched_num_patches = grid_sizes.prod(dim=1).div(merge_sizes ** 2).long()

        mm_features, mm_features_raw = self.encode_images(pixel_values, grid_sizes, merge_sizes)
        mm_features = mm_features.to(input_ids.device)
        mm_features_raw = mm_features_raw.to(input_ids.device)
        mm_features = self._get_valid_visual_tokens(mm_features, batched_num_patches, modals)

        # 3. embed text tokens
        image_selected = (input_ids == self.config.image_token_index)
        # input_ids[image_selected] = 0
        inputs_embeds = self.get_model().embed_tokens(input_ids).clone()

        num_vision_tokens = image_selected.sum()
        if mm_features.size(0) > num_vision_tokens:
            print(f"Number of mm_features ({mm_features.size(0)}) exceeds the number of image tokens ({num_vision_tokens}). Automative truncated.")
            mm_features = mm_features[:num_vision_tokens]

        # 4. replace multimodal tokens with features
        inputs_embeds[image_selected] = inputs_embeds[image_selected] * 0.0 + mm_features
        
        # 5. embed region tokens
        try:

            mask_selected = (input_ids == self.config.region_token_index)

            if mask_selected.sum() > 0:
                reshaped_features = reshape_images_to_raw_grid(mm_features_raw, grid_sizes)
                mask_additional_image_features = []
                idx = 0
                new_masks = []
                for bs in range(len(masks)):
                    flag=True
                    for ml in range(len(masks[bs])):
                        if mask_ids[idx]>=0:
                            mask_additional_image_features.append(reshaped_features[mask_ids[idx]])
                        else:
                            flag=False
                        idx+=1
                    if flag:
                        new_masks.append(masks[bs])

                mask_feats = self.get_model().region_encoder(mask_additional_image_features, new_masks)
                inputs_embeds[mask_selected] = inputs_embeds[mask_selected]*0.0 + mask_feats
            
        except Exception as e:
            print(e)


        # 6. reshape back to batched format
        C = inputs_embeds.shape[-1]
        inputs_embeds = inputs_embeds.reshape(B, -1, C)
        if attention_mask is not None:
            attention_mask = attention_mask.view(B, -1)
        if labels is not None:
            labels = labels.view(B, -1)
        if position_ids is not None:
            position_ids = position_ids.view(B, -1)

        return None, attention_mask, position_ids, past_key_values, inputs_embeds, labels

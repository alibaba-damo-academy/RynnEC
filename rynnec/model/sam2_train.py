# Adopted from https://github.com/magic-research/Sa2VA/blob/main/projects/llava_sam2/models/sam2_train.py.
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

import os.path

import torch
import torch.nn as nn

from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .utils import load_checkpoint_with_prefix, load_state_dict_to_model

BASE_DIR = 'pretrained/'


class SAM2TrainRunner(nn.Module):
    def __init__(
            self,
            cfg_path: str = "sam2_hiera_l.yaml",
            ckpt_path: str = "sam2_hiera_large.pt",
            hydra_overrides_extra=None,
            apply_postprocessing=True,
    ):
        super().__init__()

        import third_parts.sam2 # noqa: F401

        if hydra_overrides_extra is None:
            hydra_overrides_extra = []
        hydra_overrides = [
            ## Extension: LLM prompt
            "++model._target_=rynnec.model.extension.SAM2Base",
        ]

        if apply_postprocessing:
            hydra_overrides_extra = hydra_overrides_extra.copy()
            
        hydra_overrides.extend(hydra_overrides_extra)

        # Read config and init model
        cfg = compose(config_name=cfg_path, overrides=hydra_overrides)
        OmegaConf.resolve(cfg)
        sam2_model = instantiate(cfg.model, _recursive_=True)
        state_dict = load_checkpoint_with_prefix(ckpt_path)
        load_state_dict_to_model(sam2_model, state_dict)

        self.sam2_model = sam2_model

        self.hidden_dim = self.sam2_model.hidden_dim
        self.img_mean = (0.485, 0.456, 0.406)
        self.img_std = (0.229, 0.224, 0.225)

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image / 255.
        img_mean = torch.tensor(self.img_mean, dtype=image.dtype, device=image.device)[:, None, None]
        img_std = torch.tensor(self.img_std, dtype=image.dtype, device=image.device)[:, None, None]
        image -= img_mean
        image /= img_std
        return image

    def inject_language_embd(self, sam_states, language_embd, nf_nobj=None):
        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(sam_states['current_vision_feats'][:-1], sam_states['feat_sizes'][:-1])
        ]

        B = sam_states['current_vision_feats'][-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = sam_states['feat_sizes'][-1]

        if self.sam2_model.directly_add_no_mem_embed:
            # directly add no-mem embedding (instead of using the transformer encoder)
            pix_feat_with_mem = sam_states['current_vision_feats'][-1] + self.sam2_model.no_mem_embed
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        else:
            raise NotImplementedError("directly add no memory embedding is not implemented")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, _, _, low_res_masks, high_res_masks, obj_ptr, _, = self.sam2_model._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=None,
                mask_inputs=None,
                high_res_features=high_res_features,
                multimask_output=self.sam2_model._use_multimask(is_init_cond_frame=True, point_inputs=None),
                # Inject language Embed if possible
                language_embd=language_embd,
            )

        if nf_nobj is not None:
            pred_masks = low_res_masks.squeeze(1)
            pred_masks = pred_masks.unflatten(0, nf_nobj)
        else:
            pred_masks = low_res_masks
        return pred_masks

    def get_sam2_embeddings(self, images, expand_size=1):
        # Step 1: inference the backbone with the images
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            feats = self.sam2_model.forward_image(images)

        if expand_size > 1:
            # feats['vision_features'] = feats['vision_features'][:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1)
            for i, feat in enumerate(feats["backbone_fpn"]):
                feats["backbone_fpn"][i] = feat[:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1)
            for i, pos in enumerate(feats["vision_pos_enc"]):
                pos = pos[:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1)
                feats["vision_pos_enc"][i] = pos

        # Step 2: Process the features to output
        _, current_vision_feats, current_vision_pos_embeds, feat_sizes = self.sam2_model._prepare_backbone_features(feats)

        return {
            "current_vision_feats": current_vision_feats,
            "current_vision_pos_embeds": current_vision_pos_embeds,
            "feat_sizes": feat_sizes,
        }

    def forward(self, batch):
        raise NotImplementedError

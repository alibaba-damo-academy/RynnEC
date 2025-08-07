# Adopted from https://github.com/magic-research/Sa2VA/blob/main/projects/llava_sam2/models/sam2.py.
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

class SAM2(nn.Module):
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
            "++model._target_=rynnec.model.predictor.SAM2VideoPredictor",
        ]

        if apply_postprocessing:
            hydra_overrides_extra = hydra_overrides_extra.copy()
            hydra_overrides_extra += [
                # dynamically fall back to multi-mask if the single mask is not stable
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
                # "++model.binarize_mask_from_pts_for_mem_enc=true",
                # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
                # "++model.fill_hole_area=8",
            ]
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

    def inject_language_embd(self, inference_state, language_embd):
        num_frame = len(language_embd)
        num_obj = len(language_embd[0])
        mask_out = []
        for frame_idx in range(num_frame):
            frame_mask_out = []
            for obj_idx in range(num_obj):
                _language_embd = language_embd[frame_idx][obj_idx][None][None]
                _, _, out_mask_logits = self.sam2_model.add_language_embd(inference_state, frame_idx, obj_idx + 100, _language_embd)
                frame_mask_out.append(out_mask_logits)
            frame_mask_out = torch.cat(frame_mask_out, dim=1)
            mask_out.append(frame_mask_out)
        mask_out = torch.cat(mask_out, dim=0)
        return mask_out


    def language_embd_inference(self, inference_state, language_embd):
        num_frame = len(language_embd)
        num_obj = len(language_embd[0])
        mask_out = []
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for frame_idx in range(num_frame):
                frame_mask_out = []

                for obj_idx in range(num_obj):
                    _language_embd = language_embd[frame_idx][obj_idx][None][None]
                    _, _, out_mask_logits = self.sam2_model.add_language_embd(
                        inference_state,
                        frame_idx,
                        obj_idx + 100,
                        _language_embd,
                        inference=True,
                    )
                    frame_mask_out.append(out_mask_logits)
                frame_mask_out = torch.cat(frame_mask_out, dim=1)
                mask_out.append(frame_mask_out)


            mask_out = []
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_model.propagate_in_video(inference_state):
                mask_out.append(out_mask_logits)
            mask_out = torch.cat(mask_out, dim=0)
        return mask_out

    def get_sam2_embeddings(self, images):
        return self.sam2_model.init_state(images)

    def forward(self, batch):
        raise NotImplementedError

    def preprocess_image(self, image: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
        image = image / 255.

        img_mean = torch.tensor(self.img_mean, dtype=dtype, device=image.device)[:, None, None]
        img_std = torch.tensor(self.img_std, dtype=dtype, device=image.device)[:, None, None]
        image -= img_mean
        image /= img_std

        return image

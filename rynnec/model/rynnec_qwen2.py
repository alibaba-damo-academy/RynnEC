# Adopted from: https://github.com/DAMO-NLP-SG/VideoLLaMA3. 
# Adopted from: https://github.com/haotian-liu/LLaVA. 
# Below is the original copyright:
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


from typing import List, Optional, Tuple, Union, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoImageProcessor,
                          Qwen2Config, Qwen2ForCausalLM, Qwen2Model)
from transformers.generation.utils import GenerateOutput
# from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass
from transformers.utils import ModelOutput

from .loss import cross_entropy_loss, CrossEntropyLoss, DiceLoss
from .processor import Videollama3BaseProcessor
from .rynnec_arch import RynnecMetaForCausalLM, RynnecMetaModel
from .videollama3_encoder import Videollama3ImageProcessor
from rynnec.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN
from .sam2_train import SAM2TrainRunner
from .sam2 import SAM2
from .utils import genetate_video_pred_embeddings, process_video_gt_masks

CHAT_TEMPLATE = """
{%- set identifier = 'im' %}
{% for message in messages %}
    {% if message['role'] == 'stream' %}
        {% set identifier = 'stream' %}
    {% else %}
        {% set identifier = 'im' %}
    {% endif %}
    {% if message['role'] is not none %}
        {{- '<|' + identifier + '_start|>' + message['role'] + '\n' -}}
    {% endif %}
    {% if message['content'] is string %}
        {{- message['content'] + '<|' + identifier + '_end|>\n' -}}
    {% else %}
        {% for content in message['content'] %}
            {% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}
                {% if 'time' in content %}
                    {{- 'Time ' + content['time'] | round(1) | string + 's: ' -}}
                {% endif %}
                {{- image_token + '\n' -}}
            {% elif content['type'] == 'video' or 'video' in content or 'video_url' in content %}
                {% for i in range(content['num_frames']) %}
                    {% if 'timestamps' in content %}
                        {{- 'Time ' + content['timestamps'][i] | round(1) | string + 's:' -}}
                    {% endif %}
                    {% if i < content['num_frames'] - 1 %}
                        {{- image_token + ',' -}}
                    {% else %}
                        {{- image_token + '\n' -}}
                    {% endif %}
                {% endfor %}
            {% elif content['type'] == 'text' or 'text' in content %}
                {{- content['text'] -}}
            {% endif %}
        {% endfor %}
        {% if message['role'] is not none %}
            {{- '<|' + identifier + '_end|>\n' -}}
        {% endif %}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' -}}
    {% if add_think_prompt %}
        {{- '<think>\n' -}}
    {% endif %}
{% endif %}
"""
@dataclass
class CausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    ce_loss: Optional[torch.FloatTensor] = None
    mask_bce_loss: Optional[torch.FloatTensor] = None
    mask_dice_loss: Optional[torch.FloatTensor] = None
    mask_loss: Optional[torch.FloatTensor] = None


class Videollama3Qwen2Processor(Videollama3BaseProcessor):

    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    chat_template = CHAT_TEMPLATE

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_merge_size: int = 1,
        video_merge_size: int = 2,
        fps=1,
        max_frames=180,
        **kwargs
    ):
        super().__init__(image_processor, tokenizer, chat_template, **kwargs)
        self.generation_prompt = self._infer_generation_prompt()
        self.generation_prompt_ids = self.tokenizer.encode(self.generation_prompt, return_tensors="pt")
        self.generation_prompt_length = len(self.generation_prompt_ids[0])

    def _infer_generation_prompt(self):
        pseudo_message = [{"role": "user", "content": ""}]
        instruction = self.apply_chat_template(pseudo_message, tokenize=False, add_generation_prompt=True)
        conversation = self.apply_chat_template(pseudo_message, tokenize=False, add_generation_prompt=False)
        return instruction.replace(conversation, "")

    def _process_text_with_label(
        self,
        text: List[Dict],
        grid_sizes: torch.Tensor = None,
        **kwargs,
    ):
        assert kwargs.pop("return_tensors", "pt") == "pt", "Only PyTorch tensors are supported when return_labels=True."
        assert isinstance(text[0], dict), "When return_labels=True, text must be a list of messages."

        input_ids_list = []
        targets_list = []
        image_idx = 0

        for message_idx, message in enumerate(text):
            # 1. set chat template and append image tokens
            prompt = self.apply_chat_template([message], tokenize=False, add_generation_prompt=False)
            prompt_chunks = prompt.split(DEFAULT_IMAGE_TOKEN)
            prompt = []
            for chunk_idx in range(len(prompt_chunks) - 1):
                prompt.append(prompt_chunks[chunk_idx])
                thw = grid_sizes[image_idx]
                prompt.append(DEFAULT_IMAGE_TOKEN * thw.prod().long())
                image_idx += 1
            prompt.append(prompt_chunks[-1])
            prompt = "".join(prompt)

            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")[0]
            input_ids_list.append(input_ids)

            targets = torch.full_like(input_ids, IGNORE_INDEX)
            if message["role"] == "assistant" or message["role"] is None:
                targets[self.generation_prompt_length:-1] = input_ids[self.generation_prompt_length:-1].clone()

                # NOTE: mask out image tokens
                vision_mask = input_ids == self.image_token_id
                targets[vision_mask] = IGNORE_INDEX
                vision_indices = torch.nonzero(vision_mask, as_tuple=True)[0]
                targets[vision_indices + 1] = IGNORE_INDEX

                # NOTE: mask out <think> or <think>\n
                think_mask = targets == self.think_start_token_id
                targets[think_mask] = IGNORE_INDEX
                think_indices = torch.nonzero(think_mask, as_tuple=True)[0]
                newline_mask = torch.zeros_like(think_mask)
                newline_mask[think_indices + 1] = targets[think_indices + 1] == self.newline_token_id
                targets[newline_mask] = IGNORE_INDEX

            targets_list.append(targets)

        assert len(grid_sizes) == image_idx, "Number of images does not match the number of image tokens in the text."

        text_inputs = {
            "input_ids": torch.cat(input_ids_list),
            "labels": torch.cat(targets_list),
        }

        return text_inputs


class RynnecQwen2Config(Qwen2Config):
    model_type = "rynnec_qwen2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "rynnec_qwen2"


class RynnecQwen2Model(RynnecMetaModel, Qwen2Model):
    config_class = RynnecQwen2Config

    def __init__(self, config: RynnecQwen2Config):
        super(RynnecQwen2Model, self).__init__(config)

        if hasattr(config, "mm_mask_decoder"): # inference
            self.build_mask_decoder(config)
        else: # training
            if 'out_dim' not in config:
                config.out_dim = 256        

    def build_mask_decoder(self, config):
            
        # Projection layer for lisa
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True    


class RynnecQwen2ForCausalLM(Qwen2ForCausalLM, RynnecMetaForCausalLM):
    config_class = RynnecQwen2Config

    def __init__(self, config, **kwargs):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = RynnecQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        
        if hasattr(config, "training") and config.training is True:
            self.grounding_encoder = SAM2TrainRunner(ckpt_path=config.mask_decoder_model)
            config.mm_mask_decoder = True
        else:
            self.grounding_encoder = SAM2(ckpt_path=config.mask_decoder_model)
        
        self.loss_mask = CrossEntropyLoss(
            use_sigmoid=True,
            reduction='mean',
            loss_weight=2.0
        )
        self.loss_dice = DiceLoss(
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=0.5
        )

    def load_sam2_weights(self, model_path):
        sam2_model = torch.load(model_path, map_location='cpu')['model']
        prefix = "sam2_model."
        new_state_dict = {}
        for param_name in sam2_model.keys():
            new_param_name = prefix + param_name
            new_state_dict[new_param_name] = sam2_model[param_name]

        self.grounding_encoder.load_state_dict(new_state_dict, strict=False)

    def get_model(self):
        return self.model
    # NOTE: arguments are copied from transformers==4.46.3
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        # multimodal inputs
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        masks: Optional[List[torch.LongTensor]] = None,
        mask_ids = None,
        sam_images = None,
        sam_size = None,
        image2maskids = None,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        torch.cuda.empty_cache()
        if inputs_embeds is None:
            input_ids_raw = input_ids.clone()
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
                masks=masks,
                mask_ids=mask_ids
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        loss, logits = None, None
        _valid = True
        seg_valid = True

        if labels is not None: #training

            ce_loss = cross_entropy_loss(
                hidden_states=hidden_states,
                lm_head=self.lm_head,
                position_ids=position_ids,
                labels=labels,
                reduction_scope=self.config.loss_reduction_scope,
                **loss_kwargs,
            )

            if self.config.has_mask:

                hidden_states_sam = []
                hidden_states_sam.append(self.model.text_hidden_fcs[0](hidden_states))
                hidden_states_sam = torch.stack(hidden_states_sam, dim=-1).sum(dim=-1)

                bs = input_ids_raw.shape[0]
                gt_masks_list = []
                pred_masks_list = []
                mask_bce_loss = 0
                mask_dice_loss = 0
                num_masks = 0
                for i in range(bs):
                    pred_masks = []
                    pred_embeddings = []
                    input_id = input_ids_raw[i]
                    seg_token_mask = input_id[1:]==self.config.seg_token_index
                    seg_token_mask = torch.cat(
                        [
                            seg_token_mask,
                            torch.zeros((1)).bool().cuda(),
                        ],
                        dim=0,
                    )

                    pred_embedding = hidden_states_sam[i][seg_token_mask]
                    if len(pred_embedding)>0:
                        pred_embeddings.append(pred_embedding)
                    else:
                        pred_embeddings.append(hidden_states_sam[i, :1])
        
                
                    gt_masks_video = []  # FIXME: Only support one segmentation now
                    gt_mask = masks[i]
                    mask_valid = True
                    
                    if len(image2maskids[i])==0:
                        sam_images[i] = sam_images[i][:1]
                        gt_masks_video.append(torch.zeros((len(sam_images[i]), 224, 224)).to(sam_images[0].device))
                        mask_valid = False
                        
                    else:
                        for mids in image2maskids[i]:
                            for mid in mids:
                                if mid is None:
                                    gt_masks_video.append(torch.zeros((224, 224)).unsqueeze(0).to(gt_mask[0].device))
                                else:
                                    gt_masks_video.append(gt_mask[mid].unsqueeze(0))
                    frames_per_batch = [len(sam_images[i])]
                    try:
                        pred_embeddings_list_video = genetate_video_pred_embeddings(pred_embeddings, frames_per_batch)

                        # pred_embeddings_list_video, gt_masks_video = check_obj_number(pred_embeddings_list_video, gt_masks_video)

                        g_pixel_values = sam_images[i]
                        num_objs = len(pred_embeddings_list_video[0]) 
                
                        # with torch.no_grad():
            
                        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values, expand_size=num_objs)
                        language_embeddings = torch.cat(pred_embeddings_list_video, dim=0)[:, None]#.contiguous()

                        num_frames = len(pred_embeddings_list_video)
                        gt_masks_video = process_video_gt_masks(gt_masks_video, num_frames, num_objs)
                        pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))
                        
                        gt_masks = [F.interpolate(gt_mask.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gt_mask in gt_masks_video]
                        gt_masks = torch.cat(gt_masks, dim=0)
                        pred_masks = pred_masks.flatten(0, 1)

                        if not mask_valid:
                            pred_masks = pred_masks*0.0
                    
                        if len(pred_masks) != len(gt_masks):
                            # drop this data
                            print(f"Pred mask shape {pred_masks.shape} is not equal to gt_mask shape {gt_masks.shape} !!!")
                            min_num = min(len(pred_masks), len(gt_masks))
                            pred_masks = pred_masks[:min_num]
                            gt_masks = gt_masks[:min_num]
                            seg_valid = False

                        if not seg_valid or not mask_valid:
                            _scale = 0.0
                        else:
                            _scale = 1.0

                        mask_bce_loss_ = self.loss_mask(pred_masks, gt_masks) * len(pred_masks) * _scale
                        mask_dice_loss_ = self.loss_dice(pred_masks, gt_masks) * len(gt_masks) * _scale
                        mask_bce_loss += mask_bce_loss_
                        mask_dice_loss += mask_dice_loss_
                        num_masks += len(pred_masks)
                    except Exception as exp:
                        print(exp) 
                        _valid = False
            

                if num_masks>0:
                    mask_bce_loss = mask_bce_loss / num_masks
                    mask_dice_loss = mask_dice_loss / num_masks

                mask_bce_loss = self.config.bce_loss_weight * mask_bce_loss 
                mask_dice_loss = self.config.dice_loss_weight * mask_dice_loss 
                if _valid==False:
                    mask_bce_loss = mask_bce_loss * 0.0
                    mask_dice_loss = mask_dice_loss* 0.0
                
                mask_loss = mask_bce_loss + mask_dice_loss
                loss = mask_loss + ce_loss
            else:
                loss = ce_loss

        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if loss is not None:
            if self.config.has_mask:
                return CausalLMOutputWithPast(
                    loss=loss,
                    ce_loss=ce_loss.detach(),
                    mask_bce_loss=mask_bce_loss.detach(),
                    mask_dice_loss=mask_dice_loss.detach(),
                    mask_loss=mask_loss.detach(),
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
            else:
                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
        else: #infer
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    @torch.no_grad()
    def inference(
        self,
        # multimodal inputs
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        masks: Optional[List[torch.LongTensor]] = None,
        mask_ids = None,
        sam_images = None,
        sam_size = None,
        image2maskids = None,
        seg_start_idx = 0,
        **kwargs,
    ):
        outputs = self.generate(
            pixel_values=pixel_values,
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
            modals=modals,
            masks=masks,
            mask_ids=mask_ids,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **kwargs
        )

        input_ids = kwargs.pop('input_ids')
        last_hidden_state = []
        for hs in outputs.hidden_states: # round
            last_hidden_state.append(hs[-1])
        last_hidden_state = torch.cat(last_hidden_state, dim=1)

        output_ids = outputs.sequences

        concat_ids = torch.cat((input_ids, output_ids), dim=1)
        seg_token_mask = concat_ids[:, 1:] == self.config.seg_token_index

        last_hidden_state_sam = self.model.text_hidden_fcs[0](last_hidden_state)

        pred_embeddings = last_hidden_state_sam[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum() 

        if seg_token_counts>0:

            g_pixel_values = torch.cat(sam_images, dim=0).contiguous()
            num_objs = 1 #FIXME: Only support one segmentation now
            if seg_start_idx>0:
            # before start idx
                g_pixel_values_beg = g_pixel_values[:seg_start_idx+1].flip(0)
                num_frames = len(g_pixel_values_beg)
                sam_states_beg = self.grounding_encoder.get_sam2_embeddings(g_pixel_values_beg)
                pred_masks_beg = self.grounding_encoder.language_embd_inference(sam_states_beg, [pred_embeddings]*num_frames)
            else:
                pred_masks_beg = torch.zeros((1, 1, 1024, 1024)).to(pixel_values.device)
            
            if seg_start_idx<=len(g_pixel_values)-1:
                g_pixel_values_end = g_pixel_values[seg_start_idx:]
                num_frames = len(g_pixel_values_end)
                sam_states_end = self.grounding_encoder.get_sam2_embeddings(g_pixel_values_end)
                pred_masks_end = self.grounding_encoder.language_embd_inference(sam_states_end, [pred_embeddings]*num_frames)
            else:
                pred_masks_end = torch.zeros((0, 1, 1024, 1024)).to(pixel_values.device)
            
            pred_masks = torch.cat([pred_masks_beg[1:].flip(0), pred_masks_end], dim=0)

        return output_ids, pred_masks

        
    @torch.no_grad()
    def generate(
        self,
        # multimodal inputs
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        masks: Optional[List[torch.LongTensor]] = None,
        mask_ids = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        input_ids = kwargs.pop("input_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        position_ids = kwargs.pop("position_ids", None)
        past_key_values = kwargs.pop("past_key_values", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if pixel_values is not None:
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=None,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
                masks=masks,
                mask_ids=mask_ids
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs


AutoConfig.register("rynnec_qwen2", RynnecQwen2Config)
AutoModelForCausalLM.register(RynnecQwen2Config, RynnecQwen2ForCausalLM)
AutoProcessor.register(RynnecQwen2Config, Videollama3Qwen2Processor)
AutoImageProcessor.register(RynnecQwen2Config, Videollama3ImageProcessor)

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
import warnings
import shutil

import torch
from transformers import PretrainedConfig, AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, AutoProcessor

from .projector import load_mm_projector
from .videollama3_encoder import Videollama3VisionEncoderModel, Videollama3VisionEncoderConfig
from .rynnec_qwen2 import RynnecQwen2ForCausalLM, RynnecQwen2Config, Videollama3Qwen2Processor

def apply_liger_kernel_to_rynnec():
    from liger_kernel.transformers import (
        apply_liger_kernel_to_mistral,
        apply_liger_kernel_to_qwen2,
        apply_liger_kernel_to_qwen3,
        apply_liger_kernel_to_qwen3_moe,
    )
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from liger_kernel.transformers.layer_norm import LigerLayerNorm
    from .videollama3_encoder import modeling_videollama3_encoder

    apply_liger_kernel_to_mistral()
    apply_liger_kernel_to_qwen2()

    modeling_videollama3_encoder.apply_rotary_pos_emb_vision = liger_rotary_pos_emb
    modeling_videollama3_encoder.LayerNorm = LigerLayerNorm


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", **kwargs):
    if 'token' in kwargs:
        token = kwargs['token']
    else:
        token = None

    # NOTE: auto device_map by default
    # if want to put model into a single device, you can set device_map={"": "cuda:0"}
    kwargs = {"device_map": device_map, **kwargs}

    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = kwargs.pop('attn_implementation', "flash_attention_2") # default to flash_attention_2

    torch_dtype = config.torch_dtype if hasattr(config, "torch_dtype") else kwargs.pop('torch_dtype', torch.float16)

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        # NOTE: High-version Transformers will report: """ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time."""
        # kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch_dtype

    # judge model type
    model_type = config.model_type if hasattr(config, "model_type") else kwargs.pop('model_type', "rynnec_qwen2")

    # judge pretrain/finetune
    is_alignment = getattr(config, "tune_mm_mlp_adapter", False) or getattr(config, "is_alignment", False)

    # NOTE: lora/qlora model loading
    if 'lora' in model_name.lower() or 'qlora' in model_name.lower():
    # if True:
        cfg_pretrained = PretrainedConfig.from_pretrained(model_path, token=token)
        # NOTE: AutoConfig will modify `_name_or_path` property to `model_path` if `model_path` is not None.
        # cfg_pretrained = AutoConfig.from_pretrained(model_path, token=token)
        model_base = model_base if model_base is not None else cfg_pretrained._name_or_path

        # NOTE: remove qlora training quantization config 
        if hasattr(cfg_pretrained, 'quantization_config'):
            del cfg_pretrained.quantization_config
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)
        print('Loading RynnEC from base model...')

        config_raw = AutoConfig.from_pretrained(model_base)
        new_vocab_size = config.vocab_size 
        if config.vocab_size!=config_raw.vocab_size:
            config.vocab_size = config_raw.vocab_size
        config.training = False

        if 'qwen2' in model_base.lower():
            model = RynnecQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)
        else:
            model = RynnecQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)

        model.config.mask_decoder_model = "./checkpoints/sam2_hiera_large.pt"
       
        token_num, tokem_dim = new_vocab_size, model.lm_head.in_features

        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional RynnEC weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            # this is probably from HF Hub
            from huggingface_hub import hf_hub_download
            def load_from_hf(repo_id, filename, subfolder=None):
                cache_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder)
                return torch.load(cache_file, map_location='cpu')
            non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
        
        # add
        sam2_model = torch.load(model.config.mask_decoder_model, map_location='cpu')['model']
        prefix = "base_model.model.grounding_encoder.sam2_model."
        for param_name in sam2_model.keys():
            new_param_name = prefix + param_name
            if new_param_name not in non_lora_trainables.keys():
                non_lora_trainables[new_param_name] = sam2_model[param_name]

        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')


    elif model_base is not None or '-base' in model_name.lower() or is_alignment:
        # NOTE: Base/Pretrain model loading
        print('Loading RynnEC from base model...')
        cfg_pretrained = PretrainedConfig.from_pretrained(model_path, token=token)
        # NOTE: AutoConfig will modify `_name_or_path` property to `model_path` if `model_path` is not None.
        # cfg_pretrained = AutoConfig.from_pretrained(model_path, token=token)
        model_base = model_base if model_base is not None else cfg_pretrained._name_or_path

        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, token=token)

        if model_type in ['rynnec', 'rynnec_qwen2']:
            model = RynnecQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)
        else:
            model = RynnecQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)

        # NOTE; loading vision-language projector
        # * old codes for loading local mm_projector.bin
        # mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
        # mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        # model.load_state_dict(mm_projector_weights, strict=False)
        # * new codes which supports loading mm_projector.bin both offline and online 
        mm_projector_weights = load_mm_projector(model_path, token=token)
        model.load_state_dict(mm_projector_weights, strict=False)
    elif 'rynnec' in model_type:
        # NOTE: SFT model loading
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)

        if model_type in ['rynnec_qwen2']:
            model = RynnecQwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=config, **kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=config, **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, token=token)
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config, **kwargs)

    processor = None

    # if "videollama" in model_type:
    if True:
        vision_encoder = model.get_vision_encoder()
        processor = vision_encoder.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, processor, context_len

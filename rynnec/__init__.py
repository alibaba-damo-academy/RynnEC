import os
import copy
import math
import warnings
import shutil
from functools import partial

import torch
import numpy as np
from .model import load_pretrained_model
from .mm_utils import load_images, process_images, load_video, process_video, tokenizer_multimodal_token, get_model_name_from_path, KeywordsStoppingCriteria, DirectResize, sam_preprocess_batch
from .constants import NUM_FRAMES, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, MODAL_INDEX_MAP, STREAM_START_TOKEN, STREAM_END_TOKEN
from .model.rynnec_qwen2 import Videollama3Qwen2Processor

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def model_init(model_path=None, min_visual_tokens=None, max_visual_tokens=None, **kwargs):
    model_path = "Alibaba-DAMO-Academy/RynnEC-2B" if model_path is None else model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, **kwargs)

    if max_visual_tokens is not None:
        image_processor.max_tokens = max_visual_tokens
    if min_visual_tokens is not None:
        image_processor.min_tokens = min_visual_tokens

    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    processor = Videollama3Qwen2Processor(image_processor, tokenizer)

    return model, processor


def mm_infer(images_or_videos, vlprocessor, instruct, model, tokenizer, modal='video', **kwargs):

    mask_ids = kwargs.pop('mask_ids', None)
    masks = kwargs.pop('masks', None)
    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
        images = images_or_videos
        timestamps = None
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
        images, timestamps = images_or_videos
    elif modal == 'text':
        modal_token = ''
    else:
        raise ValueError(f"Unsupported modal: {modal}")


    # 1. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        messages = [{'role': 'user', 'content': instruct}]
    elif isinstance(instruct, list):
        messages = copy.deepcopy(instruct)
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")

    if all(not modal_token in message["content"] for message in messages):
        warnings.warn(f"Image tag not found in the conversation, add it automatically at the beginning!")
        messages[0]["content"] = modal_token + messages[0]["content"]

    converted_messages = []
    for message in messages:
        chunks = message["content"].split(modal_token)
        converted_messages.append({
            "role": "user",
            "content": []
        })

        for chunk_idx in range(1, 2 * len(chunks)):
            if chunk_idx % 2 == 1:
                chunk = chunks[chunk_idx // 2].strip()
                converted_messages[-1]["content"].append({"type": "text",  "text": chunk}) if chunk else None
            else:
                if modal == 'image':
                    converted_messages[-1]["content"].append({"type": "image"})
                elif modal == 'video':
                    converted_messages[-1]["content"].append({"type": "video", "num_frames": len(images), "time": timestamps})

    messages = converted_messages

    system_message = []

    image_downsampling = kwargs.get('image_downsampling', model.config.spatial_merge_size)
    # TODO: attention mask?
    messages = system_message + messages
    data_dict = vlprocessor(
        images=images,
        text=messages,
        merge_size=image_downsampling,
        return_labels=True,
        return_tensors="pt",
    )

    torch_dtype = model.config.torch_dtype if hasattr(model.config, "torch_dtype") else torch.float16

    # images = [x.to(torch_dtype).cuda(non_blocking=True) for x in data_dict["images"]]
    # grid_thws = [x.cuda(non_blocking=True) for x in data_dict["grid_thws"]]

    # 3. generate response according to visual signals and prompts. 
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, data_dict["input_ids"].unsqueeze(0))

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 1.0)
    top_p = kwargs.get('top_p', 0.9 if do_sample else 1.0)
    top_k = kwargs.get('top_k', 20 if do_sample else 50)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)

    data_dict["modals"] = [modal]
    data_dict = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}
    if "pixel_values" in data_dict:
        data_dict["modals"] = data_dict["modals"] * len(data_dict["grid_sizes"])
        data_dict["pixel_values"] = data_dict["pixel_values"].to(torch.bfloat16)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=data_dict["input_ids"].unsqueeze(0).cuda(),
            pixel_values=data_dict["pixel_values"],
            grid_sizes=data_dict["grid_sizes"],
            merge_sizes=data_dict["merge_sizes"],
            modals=data_dict["modals"],
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
            masks=[masks],
            mask_ids=mask_ids
        )

    outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return outputs

def mm_infer_segmentation(images_or_videos, vlprocessor, instruct, model, tokenizer, modal='video', seg_start_idx=0, **kwargs):

    image2maskids = kwargs.get('image2maskids', [])
    img_size=1024
    sam_transform = DirectResize(img_size)


    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
        images = images_or_videos
        timestamps = None
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
        images, timestamps = images_or_videos
    elif modal == 'text':
        modal_token = ''
    else:
        raise ValueError(f"Unsupported modal: {modal}")


    sam_images = []
    sam_size = None
    if len(images)>0:
        for image in images:
            sam_image = sam_transform.apply_image(np.array(image))
            sam_images.append(sam_image)
            if sam_size is None:
                sam_size = sam_image.shape[:2] 
        sam_images = np.array(sam_images)
        sam_images = torch.from_numpy(sam_images).permute(0, 3, 1, 2).contiguous()
        sam_images = sam_preprocess_batch(sam_images)


    # 1. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        messages = [{'role': 'user', 'content': instruct}]
    elif isinstance(instruct, list):
        messages = copy.deepcopy(instruct)
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")

    if all(not modal_token in message["content"] for message in messages):
        warnings.warn(f"Image tag not found in the conversation, add it automatically at the beginning!")
        messages[0]["content"] = modal_token + messages[0]["content"]

    converted_messages = []
    for message in messages:
        chunks = message["content"].split(modal_token)
        converted_messages.append({
            "role": "user",
            "content": []
        })

        for chunk_idx in range(1, 2 * len(chunks)):
            if chunk_idx % 2 == 1:
                chunk = chunks[chunk_idx // 2].strip()
                converted_messages[-1]["content"].append({"type": "text",  "text": chunk}) if chunk else None
            else:
                if modal == 'image':
                    converted_messages[-1]["content"].append({"type": "image"})
                elif modal == 'video':
                    converted_messages[-1]["content"].append({"type": "video", "num_frames": len(images), "time": timestamps})

    messages = converted_messages

    system_message = []

    image_downsampling = kwargs.get('image_downsampling', model.config.spatial_merge_size)
    # TODO: attention mask?
    messages = system_message + messages
    data_dict = vlprocessor(
        images=images,
        text=messages,
        merge_size=image_downsampling,
        return_labels=True,
        return_tensors="pt",
    )

    torch_dtype = model.config.torch_dtype if hasattr(model.config, "torch_dtype") else torch.float16

    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, data_dict["input_ids"].unsqueeze(0))

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 1.0)
    top_p = kwargs.get('top_p', 0.9 if do_sample else 1.0)
    top_k = kwargs.get('top_k', 20 if do_sample else 50)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)

    torch_dtype = model.config.torch_dtype if hasattr(model.config, "torch_dtype") else torch.float16

    data_dict["modals"] = [modal]
    data_dict = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}
    if "pixel_values" in data_dict:
        data_dict["modals"] = data_dict["modals"] * len(data_dict["grid_sizes"])
        data_dict["pixel_values"] = data_dict["pixel_values"].to(torch.bfloat16)

    with torch.inference_mode():
        output_ids, pred_masks = model.inference(
            input_ids=data_dict["input_ids"].unsqueeze(0).cuda(),
            pixel_values=data_dict["pixel_values"],
            grid_sizes=data_dict["grid_sizes"],
            merge_sizes=data_dict["merge_sizes"],
            modals=data_dict["modals"],
            sam_images=[sam_images],
            sam_size=[sam_size],
            image2maskids=[image2maskids],
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
            seg_start_idx=seg_start_idx
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    pred_masks_sigmoid = pred_masks.sigmoid()>0.5

    return outputs, pred_masks_sigmoid

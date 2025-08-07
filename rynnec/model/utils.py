import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import logging
from huggingface_hub import hf_hub_download
import functools
from typing import Callable, Optional

def process_video_gt_masks(gt_masks, num_frames, num_objs):
    gt_masks_processed = []
    for i in range(num_frames):
        for j in range(num_objs):
            gt_masks_processed.append(gt_masks[j*num_frames+i])
    return gt_masks_processed

def load_checkpoint_with_prefix(filename, prefix=None, map_location='cpu', logger='current'):
    HF_HUB_PREFIX = 'hf-hub:'
    if filename.startswith(HF_HUB_PREFIX):
        model_id = filename[len(HF_HUB_PREFIX):]
        filename = hf_hub_download(model_id, 'pytorch_model.bin')

    checkpoint = torch.load(filename, map_location=map_location)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if not prefix:
        return state_dict
    if not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    state_dict = {
        k[prefix_len:]: v
        for k, v in state_dict.items() if k.startswith(prefix)
    }

    assert state_dict, f'{prefix} is not in the pretrained model'
    return state_dict


def load_state_dict_to_model(model, state_dict,  logger='current'):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict)
    if missing_keys:
        raise RuntimeError()
    if unexpected_keys:
        raise RuntimeError()

def genetate_video_pred_embeddings(pred_embeddings_list, frames_per_batch):
    assert len(pred_embeddings_list) == len(frames_per_batch), \
    f"Lengths do not match: len(pred_embeddings_list)={len(pred_embeddings_list)}, len(frames_per_batch)={len(frames_per_batch)}"
    
    pred_embeddings_list_video = []
    for pred_embedding_batch, frame_nums in zip(pred_embeddings_list, frames_per_batch):
        pred_embeddings_list_video += [pred_embedding_batch] * frame_nums
    return pred_embeddings_list_video


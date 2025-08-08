import argparse
import sys
sys.path.append('./')
import re

import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
import math

from rynnec.constants import REGION_TOKEN
from rynnec.mm_utils import annToMask, load_video, load_images



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class Cog_Dataset(Dataset):
    def __init__(self, video_folder, data_list, data_type=None):
        self.video_folder = video_folder
        self.data_list = data_list
        self.data_type = data_type
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_folder = self.video_folder
        data_dict = self.data_list[idx]
        conversation = data_dict["conversations"]
        video_file_raw = data_dict["video"]

        video_root = data_dict['video_id']
        video_file = [os.path.join(video_root, vf) for vf in video_file_raw]
        
        if all(not "<video>" in sentence["value"] for sentence in conversation):
            conversation[0]["value"] = "<video>" + conversation[0]["value"]

        
        masks = []
        mask_nums = []
        if 'masks' in data_dict and data_dict['masks'] is not None and len(data_dict['masks']) > 0 \
            and "mask_ids" in data_dict and data_dict['mask_ids'] is not None and len(data_dict['mask_ids']) > 0 \
            and '<region>' in conversation[0]['value']:

            mask_ids = data_dict["mask_ids"]
            if 'height' in data_dict:
                h = data_dict['height']
                w = data_dict['width']
            else:
                h = None
                w = None
            mask_ids_first = []
            ids = 0
            for ann in data_dict['masks']:
                mask_num = 0
                for k in ann.keys():
                    mask = annToMask(ann[k], h, w)
                    masks.append(mask)
                    mask_ids_first.append(mask_ids[ids])
                    mask_num += 1
                    ids += 1
                    
                mask_nums.append(mask_num)
            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)
            
            conv_i = 0
            for midx in range(len(mask_nums)):
                conversation[conv_i]['value'] = conversation[conv_i]['value'].replace('<region>', "["+REGION_TOKEN*mask_nums[midx]+"]", 1)
        else:
            masks = None
            mask_ids = None
            mask_ids_first = None
            

        images = []
        for vf in video_file:
            images+=load_images(os.path.join(data_folder, vf))
        timestamps = data_dict['timestamps']
        
        for conv in conversation:
            if conv["from"] == "human":
                question = conv["value"]
            else:
                answer = conv["value"]

        type_ = data_dict.get("type", 'counting')
        if isinstance(type_, list):
            type_ = type_[0]
        return {
            'idx': idx,
            'video': [images, timestamps],
            'masks': masks,
            'question': question,
            'mask_ids': mask_ids,
            'answer': answer,
            'types': type_.lower(),
            'class_name': data_dict.get("class_name", None) # only for object cognition, except for counting
        }

def collate_fn(batch):
    idx = [x['idx'] for x in batch]
    vid = [x['video'] for x in batch]
    msk = [x['masks'] for x in batch]
    qs = [x['question'] for x in batch]
    mid = [x['mask_ids'] for x in batch]
    ans = [x['answer'] for x in batch]
    tps = [x['types'] for x in batch]
    clsn = [x['class_name'] for x in batch]
    return idx, vid, msk, qs, mid, ans, tps, clsn


def build_cog_dataloader(question_file, video_folder, num_chunks=1, chunk_idx=0, batch_size=1, num_workers=8, distributed=False, **kwargs):
    # convert parquet to json
    questions = json.load(open(question_file))
    questions = get_chunk(questions, num_chunks, chunk_idx)
    dataset = Cog_Dataset(video_folder, questions)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, sampler=sampler)
    return dataloader
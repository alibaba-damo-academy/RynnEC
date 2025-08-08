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

from rynnec.mm_utils import annToMask, load_video, load_images



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class Seg_Dataset(Dataset):
    def __init__(self, video_folder, data_list, data_type=None, only_mask_img = True):
        self.video_folder = video_folder
        self.data_list = data_list
        self.data_type = data_type
        self.only_mask_img = only_mask_img
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_folder = self.video_folder
        data = self.data_list[idx]
        video_root = data["video_id"]
        instruction = data['conversations'][0]['value']
        data["mask_ids"] = [mid for mid in data["mask_ids"]]
        video_file = data["video"]
        task_type = data['type']
        

        masks = []
        mask_nums = []
        maskid = 0

        if 'masks' in data and data['masks'] is not None:
            mask_ids = data["mask_ids"]
            if 'height' in data:
                h = data['height']
                w = data['width']
            else:
                h = None
                w = None

            if isinstance(data['masks'], str):
                masks_ = json.load(open(data['masks']))
            else:
                masks_= data['masks']
            for ann in masks_:
                for k in ann.keys():
                    mask = annToMask(ann[k], h, w)
                    masks.append(mask)
                    maskid+=1

                mask_nums.append(len(ann.keys()))
            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)
        else:
            masks = None
            mask_ids = None

        if self.only_mask_img:
            video_file = [video_file[i] for i in mask_ids]
            images = []
            for vf in video_file:
                images+=load_images(os.path.join(data_folder, video_root, vf))
            timestamps = data['timestamps']

            gt_masks = masks

        else:
            images = []
            for vf in video_file:
                images+=load_images(os.path.join(data_folder, video_root, vf))
            timestamps = data['timestamps']
            
            gt_masks = torch.zeros((len(images), images[0].height, images[0].width))
            for i, mid in enumerate(mask_ids):
                gt_masks[mid] = masks[i]
        
        
        return {
            'idx': idx,
            'video': [images, timestamps],
            'masks': gt_masks,
            'instruction': instruction,
            'type': task_type,
            'mask_ids': torch.tensor(mask_ids)
        }

def collate_fn(batch):
    idx = [x['idx'] for x in batch]
    vid = [x['video'] for x in batch]
    msk = [x['masks'] for x in batch]
    ins = [x['instruction'] for x in batch]
    typ = [x['type'] for x in batch]
    maskids = [x['mask_ids'] for x in batch]
    return idx, vid, msk, ins, typ, maskids


def build_seg_dataloader(question_file, video_folder, only_mask_img, num_chunks=1, chunk_idx=0, batch_size=1, num_workers=8, distributed=False):
    # convert parquet to json
    questions = json.load(open(question_file))
    questions = get_chunk(questions, num_chunks, chunk_idx)
    dataset = Seg_Dataset(video_folder, questions, only_mask_img=only_mask_img)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, sampler=sampler)
    return dataloader
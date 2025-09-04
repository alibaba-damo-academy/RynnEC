import argparse
import sys
sys.path.append('./')
import re

import copy 
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
import math
from tqdm import tqdm
from torchvision.transforms import v2
from .utils import annToMask, load_images, load_video

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class Bbox_Dataset(Dataset):
    def __init__(self, data_list, data_type=None):
        self.data_list = data_list
        self.data_type = data_type
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]
    
def collate_fn(batch):
    data = [x for x in batch]
    return data
  
def build_bbox_dataloader(question_file, num_chunks=1, chunk_idx=0, batch_size=1, num_workers=8, distributed=False):
    # convert parquet to json
    questions = json.load(open(question_file))
    questions = get_chunk(questions, num_chunks, chunk_idx)
    dataset = Bbox_Dataset(questions)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, sampler=sampler)
    return dataloader


    
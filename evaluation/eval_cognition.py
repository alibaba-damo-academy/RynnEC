import math
import os
import argparse
import json
import copy
import warnings
from tqdm import tqdm
import sys
sys.path.append('./')

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from pycocotools import mask as maskUtils
from rynnec import disable_torch_init, model_init, mm_infer
from rynnec.mm_utils import load_video, load_images, annToMask
from rynnec.constants import (IGNORE_INDEX, MODAL_INDEX_MAP,
    NUM_FRAMES, DEFAULT_IMAGE_TOKEN, STREAM_MAX_FRAMES,
    STREAM_DOWNSAMPLING, STREAM_FPS, STREAM_IMAGE_SIZE,
    STREAM_START_TOKEN, STREAM_END_TOKEN, REGION_TOKEN)

from .metrics import calculate_score
from .utils import postprocess_prop_result, save_results, postprocess_spatial_result

NUM_FRAMES = 16


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class Prop_Dataset(Dataset):
    def __init__(self, video_folder, data_list, processor, data_type=None):
        self.video_folder = video_folder
        self.data_list = data_list
        self.processor = processor
        self.data_type = data_type
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_folder = self.video_folder
        data_dict = self.data_list[idx]
        conversation = copy.deepcopy(data_dict["conversations"])
        video_file_raw = data_dict["video"]

        video_root = data_dict['video_id']
        video_file = [os.path.join(video_root, vf) for vf in video_file_raw]
        
        if all(not "<video>" in sentence["value"] for sentence in conversation):
            warnings.warn(f"Video tag not found in the conversation, add it automatically at the beginning!")
            conversation[0]["value"] = "<video>" + conversation[0]["value"]

        
        masks = []
        mask_nums = []
        if 'masks' in data_dict and data_dict['masks'] is not None and len(data_dict['masks']) > 0 \
            and "mask_ids" in data_dict and data_dict['mask_ids'] is not None and len(data_dict['mask_ids']) > 0 \
            and '<region>' in conversation[0]['value']:
            try:
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
                    # first = True # for demo
                    for k in ann.keys():
                        mask = annToMask(ann[k], h, w)
                        masks.append(mask)
                        mask_ids_first.append(mask_ids[ids])
                        mask_num += 1
                        # first = False
                        ids += 1
                        
                    mask_nums.append(mask_num)
                masks = np.stack(masks, axis=0)
                masks = torch.from_numpy(masks)
                
                conv_i = 0
                for midx in range(len(mask_nums)):
                    # while '<region>' not in conversation[conv_i]['value']:
                    #     conv_i+=1
                    conversation[conv_i]['value'] = conversation[conv_i]['value'].replace('<region>', "["+REGION_TOKEN*mask_nums[midx]+"]", 1)
            except Exception as e:
                print(f"Data {idx} Error: {e}")
                masks = None
                mask_ids = None
        else:
            masks = None
            mask_ids = None
            mask_ids_first = None
            
        if isinstance(video_file, list) and len(video_file) == 1:
            video_file = os.path.join(data_folder, video_file[0])
            images, timestamps = load_video(video_file, fps=1, max_frames=768, frame_ids=None)
        elif isinstance(video_file, list) and len(video_file)>1: #images
            images = []
            for vf in video_file:
                images+=load_images(os.path.join(data_folder, vf))
            timestamps = data_dict['timestamps']
    
        else:
            raise ValueError(f"Unsupported video format: {video_file}")
        
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
            'class_name': data_dict.get("class_name", None)
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


def build_prop_loader(args, processor, distributed=False):
    # convert parquet to json
    questions = json.load(open(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = Prop_Dataset(args.video_folder, questions, processor)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, sampler=sampler)
    return dataloader

def save_results(data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path.endswith(".json"):
        with open(save_path, "w") as f:
            json.dump(data, f, indent=4)
    elif save_path.endswith(".jsonl"):
        with open(save_path, "w") as f:
            for info in data:
                f.write(json.dumps(info) + "\n")
    else:
        raise ValueError("Unsupported file format.")
    print(f"Answer saved at:{save_path}")

def run_inference(args):
    distributed = os.getenv('WORLD_SIZE', '1') > '1'
    if distributed:
        dist.init_process_group(backend="gloo")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        disable_torch_init()
        model, processor = model_init(args.model_path, device_map={"": f"cuda:{local_rank}"})
    else:
        local_rank = 0
        global_rank = 0
        disable_torch_init()
        model, processor = model_init(args.model_path)
    
    val_loader = build_prop_loader(args, processor, distributed)
    
    results = []
    for i, (idx, video, masks_, questions, mask_ids, answers, types, class_name) in enumerate(tqdm(val_loader, desc=f"Rank {global_rank}", total=len(val_loader), position=local_rank)):
        idx = idx[0]
        video_tensor = video[0]
        masks = masks_[0]
        question = questions[0]
        mask_ids = mask_ids[0]
        answer = answers[0]
        type_ = types[0]
        class_name = class_name[0]
        
        try:
            output = mm_infer(
                video_tensor,
                processor,
                question,
                model=model,
                tokenizer=processor.tokenizer,
                do_sample=False,
                modal='video',
                masks = masks.cuda() if masks is not None else None,
                mask_ids = mask_ids
            )
        except Exception as e:
            print(f"Data {idx} Error: {e}")
            output = ''
        # print(output)
        record = {
            'idx': idx,
            'Question': question,
            'Answer': answer,
            'pred': output,
            'type': type_,
            'class_name': class_name
        }
        try:
            score = calculate_score(record)
        except Exception as e:
            print(f"Data {idx} record {record} Error: {e}")
            score = 0
        record['score'] = score
        results.append(record)
        # except Exception as e:
        #     print(f"Data {i} Error: {e}")
            

    if distributed:
        torch.cuda.empty_cache()
        gathered_results = [None for _ in range(dist.get_world_size())]
        dist.gather_object(
            obj=results,
            object_gather_list=gathered_results if global_rank == 0 else None,
            dst=0,
        )
        if global_rank == 0:
            print("\n" * dist.get_world_size())
            results = sum(gathered_results, [])
            if args.task_type == 'property':
                results = postprocess_prop_result(results)
            elif args.task_type == 'spatial':
                results = postprocess_spatial_result(results)
            save_results(results, args.output_file)
        dist.destroy_process_group()
    else:
        results = postprocess_prop_result(results)
        save_results(results, args.output_file)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='Model path', required=True)
    parser.add_argument('--video_folder', help='Directory containing video files.', default='/mnt')
    parser.add_argument('--question_file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output_file', help='Directory to save the model results JSON.', default='./evaluate_test/object_properties_cognition_combined.json')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--task_type", type=str, default='property')
    args = parser.parse_args()

    run_inference(args)

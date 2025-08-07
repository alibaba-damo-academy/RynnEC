import argparse
import sys
sys.path.append('./')
import re

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from rynnec import disable_torch_init, model_init, mm_infer_segmentation
from rynnec.mm_utils import annToMask, load_video, load_images
import json
import numpy as np
import os
import math
from tqdm import tqdm
from torchvision.transforms import v2
from .utils import postprocess_seg_result, save_results
from .metrics import calculate_iou, db_eval_boundary


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class Mask_Dataset(Dataset):
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
            if isinstance(video_file, list) and len(video_file) == 1 and 'timestamps' not in data:
                video_file = os.path.join(data_folder, video_root, video_file[0])
                images, timestamps = load_video(video_file, fps=self.data_args.fps, max_frames=self.data_args.max_frames, frame_ids=None)
            elif isinstance(video_file, list): #images
                images = []
                for vf in video_file:
                    images+=load_images(os.path.join(data_folder, video_root, vf))
                timestamps = data['timestamps']
            
            else:
                raise ValueError(f"Unsupported video format: {video_file}")

            gt_masks = masks

        else:
            if isinstance(video_file, list) and len(video_file) == 1 and 'timestamps' not in data:
                video_file = os.path.join(data_folder, video_root, video_file[0])
                images, timestamps = load_video(video_file, fps=self.data_args.fps, max_frames=self.data_args.max_frames, frame_ids=None)
            elif isinstance(video_file, list): #images
                images = []
                for vf in video_file:
                    images+=load_images(os.path.join(data_folder, video_root, vf))
                timestamps = data['timestamps']
            
            else:
                raise ValueError(f"Unsupported video format: {video_file}")

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


def build_eval_dataloader(args, distributed=False):
    # convert parquet to json
    questions = json.load(open(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = Mask_Dataset(args.video_folder, questions, only_mask_img=args.only_mask_img)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, sampler=sampler)
    return dataloader

    

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

    model.to(torch.bfloat16)
    
    val_loader = build_eval_dataloader(args, distributed)
    
    results = []
    for i, (idx, video, masks_, instruction, typ, mask_ids) in enumerate(tqdm(val_loader, desc=f"Rank {global_rank}", total=len(val_loader), position=local_rank)):
        idx = idx[0]
        video_tensor = video[0]
        gt_masks = masks_[0]
        instruction = instruction[0]
        type_ = typ[0]

        mask_ids = mask_ids[0]
        
        
        try:
            output, masks = mm_infer_segmentation(
                video_tensor,
                processor,
                instruction,
                model=model,
                tokenizer=processor.tokenizer,
                do_sample=False,
                modal='video',
            )

            t, c = masks.shape[0], masks.shape[1]
            h, w = gt_masks.shape[1], gt_masks.shape[2]
            masks = v2.Resize([h,w])(masks)
            masks = masks.squeeze(1)

            record = {
                'idx': idx,
                'instruction': instruction,
                'type': type_
            }

            j = calculate_iou(masks, gt_masks.to(masks)).item()
            record['j'] = j
            f = db_eval_boundary(masks[mask_ids].cpu().detach().numpy(), gt_masks[mask_ids].cpu().detach().numpy()).mean()
            record['f'] = f
            results.append(record)
        except Exception as e:
            print(f"Data {i} Error: {e}")
            

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
            results = postprocess_seg_result(results)
            save_results(results, args.output_file)
        dist.destroy_process_group()
    else:
        results = postprocess_seg_result(results)
        save_results(results, args.output_file)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--video_folder', help='Directory containing video files.', default='./data')
    parser.add_argument('--question_file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output_file', help='Directory to save the model results JSON.', default='visualization/output.json')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--only_mask_img", action='store_true')
    args = parser.parse_args()
    print(args)

    run_inference(args)



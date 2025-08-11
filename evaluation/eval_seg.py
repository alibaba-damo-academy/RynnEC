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

from benchmark.dataloader import build_seg_dataloader
from benchmark.utils import postprocess_seg_result, save_results
from benchmark.metrics import calculate_iou, db_eval_boundary

    

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
    
    val_loader = build_seg_dataloader(args.question_file, args.video_folder, args.only_mask_img, args.num_chunks, args.chunk_idx, args.batch_size, args.num_workers, distributed=distributed)
    
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



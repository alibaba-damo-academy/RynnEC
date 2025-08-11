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


from rynnec import disable_torch_init, model_init, mm_infer

from benchmark.dataloader import build_cog_dataloader
from benchmark.metrics import calculate_score
from benchmark.utils import postprocess_prop_result, save_results, postprocess_spatial_result


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
    
    val_loader = build_cog_dataloader(args.question_file, args.video_folder, args.num_chunks, args.chunk_idx, args.batch_size, args.num_workers, distributed=distributed)
    
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

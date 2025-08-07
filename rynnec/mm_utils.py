# Adopted from: https://github.com/DAMO-NLP-SG/VideoLLaMA3. 
# Below is the original copyright:
#    Copyright 2025 The VideoLLaMA3 team, Alibaba Group
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

import ast
import os
import re
import math
import base64
import traceback
from io import BytesIO
from typing import Optional

import torch
import torchvision.transforms.functional as VF
import torch.nn.functional as F
import numpy as np
from transformers import StoppingCriteria

import cv2
import imageio
import ffmpeg
from PIL import Image
from decord import VideoReader, cpu

from .constants import NUM_FRAMES, MAX_FRAMES, NUM_FRAMES_PER_SECOND, MODAL_INDEX_MAP, DEFAULT_IMAGE_TOKEN
from pycocotools import mask as maskUtils

from torchvision.transforms.functional import resize, to_pil_image  # type: ignore


class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))

def sam_preprocess_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize pixel values and pad to square input for a batch of images.
    
    Args:
        images (torch.Tensor): A batch tensor of shape [N, C, H, W].
    
    Returns:
        torch.Tensor: A batch tensor with normalized and padded images 
                      (shape: [N, C, 1024, 1024]).
    """
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(1, -1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(1, -1, 1, 1)
    img_size = 1024
        
    # Normalize colors
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh)) 
    return x
 

def sam_preprocess(x: torch.Tensor) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
        
    # Normalize colors
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x
 

def reshape_images_to_raw_grid(mm_features_raw, grid_thws):
    start_idx=0
    reshaped_features = []
    # for thw_group in grid_thws:
    for tensor_thw in grid_thws:
        # for tensor_thw in thw_group:
            t, H, W = tensor_thw.squeeze().tolist()
            num_elements = H * W
            for i in range(t):
                split_tensor = mm_features_raw[start_idx:start_idx + num_elements].view(H, W, -1)
                reshaped_features.append(split_tensor)

                start_idx += num_elements

    assert len(mm_features_raw)==start_idx
    return reshaped_features

def annToMask(mask_ann, h=None, w=None):
    if isinstance(mask_ann, list):
        rles = maskUtils.frPyObjects(mask_ann, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, h, w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask

def chunk_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def grid_divide(image, cell_size):
    """
    Divides an image into grid of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        cell_size (int): The size of each cell.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    grid = []
    width, height = image.size
    for i in range(0, height, cell_size):
        row = []
        for j in range(0, width, cell_size):
            box = (j, i, j + cell_size, i + cell_size)
            row.append(image.crop(box))
        grid.append(row)

    return grid


def load_images(image_path):
    if isinstance(image_path, str) and os.path.isfile(image_path):
        # images = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)]
        images = [Image.open(image_path).convert('RGB')]
    elif isinstance(image_path, str) and os.path.isdir(image_path):
        # images = [cv2.cvtColor(cv2.imread(os.path.join(image_path, f)), cv2.COLOR_BGR2RGB) for f in sorted(os.listdir(image_path))]
        images = [Image.open(os.path.join(image_path, f)).convert('RGB') for f in sorted(os.listdir(image_path))]
    elif isinstance(image_path, list) and isinstance(image_path[0], str):
        # images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in image_path]
        images = [Image.open(f).convert('RGB') for f in image_path]
    elif isinstance(image_path, list) and isinstance(image_path[0], Image.Image):
        images = image_path
    elif isinstance(image_path, Image.Image):
        images = [image_path]
    else:
        raise ValueError(f"Unsupported image path type: {image_path}")

    return images


def process_pad_image(image, padding_value=(0, 0, 0)):
    image = expand2square(image, padding_value)

    return [image]


def find_closest_aspect_ratio(src_ratio, tgt_ratios, ori_size, tgt_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = ori_size[0] * ori_size[1]
    for ratio in tgt_ratios:
        tgt_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(src_ratio - tgt_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * tgt_size[0] * tgt_size[1] * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def process_dynamic_image(image, image_size=384, use_thumbnail=True):
    # Grid Params:
    min_num = 1
    max_num = 12

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    ori_size = image.size
    aspect_ratio = ori_size[0] / ori_size[1]

    # calculate the existing image aspect ratio
    tgt_ratios = []
    for n in range(min_num, max_num + 1):
        tgt_ratios.extend([(i, j) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num])
    tgt_ratios = set(tgt_ratios)
    tgt_ratios = sorted(tgt_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    tgt_ratio = find_closest_aspect_ratio(aspect_ratio, tgt_ratios, ori_size, image_size)

    # resize the image to the target size
    tgt_width = image_size[0] * tgt_ratio[0]
    tgt_height = image_size[1] * tgt_ratio[1]
    resized_img = image.resize((tgt_width, tgt_height))

    # NOTE: internvl2 style split the image into one column grids
    # num_grids = tgt_ratio[0] * tgt_ratio[1]
    # grid_images = []
    # for i in range(num_grids):
    #     box = (
    #         (i %  tgt_ratio[0]) * image_size[0],
    #         (i // tgt_ratio[0]) * image_size[1],
    #         (i %  tgt_ratio[0] + 1) * image_size[0],
    #         (i // tgt_ratio[0] + 1) * image_size[1],
    #     )
    #     # crop out the grid image
    #     grid_images.append(resized_img.crop(box))
    # assert len(grid_images) == num_grids
    # grid_images = [grid_images]

    # NOTE: eager implementation
    # num_grids = tgt_ratio[0] * tgt_ratio[1]
    # sub_grid_images = []
    # tmp_grid_images = []
    # for i in range(num_grids):
    #     box = (
    #         (i %  tgt_ratio[0]) * image_size[0],
    #         (i // tgt_ratio[0]) * image_size[1],
    #         (i %  tgt_ratio[0] + 1) * image_size[0],
    #         (i // tgt_ratio[0] + 1) * image_size[1],
    #     )
    #     tmp_grid_images.append(resized_img.crop(box))

    #     if (i + 1) % tgt_ratio[0] == 0:
    #         sub_grid_images.append(tmp_grid_images)
    #         tmp_grid_images = []

    image_grid = grid_divide(resized_img, image_size[0])

    if use_thumbnail:
        thumbnail_img = image.resize((image_size[0], image_size[1]))
        image_grid = [[thumbnail_img]] + image_grid

    return image_grid


def process_highres_image(image_path, image_size=384, use_thumbnail=True, padding_value=(0, 0, 0)):
    # Grid Params:
    grid_width = [1, 2, 3]
    grid_width_real = [x * image_size for x in grid_width]

    longest_side = max(image.size)
    fit_grid_width_real = [x for x in grid_width_real if x >= longest_side]
    if len(fit_grid_width_real) == 0:
        select_size = max(grid_width_real)
    else:
        select_size = min(fit_grid_width_real)

    image_padded = expand2square(image, padding_value)
    image_padded = image_padded.resize((select_size, select_size))
    image_grid = grid_divide(image_padded, image_size)

    if use_thumbnail:
        thumbnail_img = image.resize((image_size, image_size))
        image_grid = [[thumbnail_img]] + image_grid

    return image_grid


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def process_anyres_image(image, image_size=384, use_thumbnail=True, padding_value=(0, 0, 0)):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    # Grid Params:
    possible_grids = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    possible_resolutions = [(x * image_size, y * image_size) for x, y in possible_grids]

    best_resolution = select_best_resolution(image.size, possible_resolutions)

    # resize and padding image
    nw, nh = best_resolution
    ow, oh = image.size

    scale_factor = min(nw / ow, nh / oh)
    new_size = (int(ow * scale_factor), int(oh * scale_factor))

    image_padded = Image.new("RGB", (nw, nh), padding_value)
    image_padded.paste(image.resize(new_size), ((nw - new_size[0]) // 2, (nh - new_size[1]) // 2))

    image_grid = grid_divide(image_padded, image_size)

    if use_thumbnail:
        thumbnail_img = image.resize((image_size, image_size))
        image_grid = [[thumbnail_img]] + image_grid

    return image_grid


def process_adares_image(image_path, image_size=384, use_thumbnail=True):
    # Grid Params:
    min_num = 1
    max_num = 12

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    ori_size = image.size
    aspect_ratio = ori_size[0] / ori_size[1]

    # calculate the existing image aspect ratio
    tgt_ratios = []
    for n in range(min_num, max_num + 1):
        tgt_ratios.extend([(i, j) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num])
    tgt_ratios = set(tgt_ratios)
    possible_resolutions = [(x * image_size[0], y * image_size[1]) for x, y in tgt_ratios]

    # find the most possible resolution
    best_resolution = select_best_resolution(ori_size, possible_resolutions)

    # resize the image to the target size
    resized_img = image.resize((best_resolution[0], best_resolution[1]))

    image_grid = grid_divide(resized_img, image_size[0])

    if use_thumbnail:
        thumbnail_img = image.resize((image_size[0], image_size[1]))
        image_grid = [[thumbnail_img]] + image_grid

    return image_grid


def process_images(image_path, processor, aspect_ratio='pad', image_size=384, use_thumbnail=True):
    images = load_images(image_path)

    padding_value = tuple(int(x*255) for x in processor.image_mean)

    image_grids = []
    for image in images:
        if aspect_ratio == 'pad':
            image_grid = process_pad_image(image, padding_value=padding_value)
        elif aspect_ratio == 'dynamic':
            image_grid = process_dynamic_image(image, image_size=image_size, use_thumbnail=use_thumbnail)
        elif aspect_ratio == 'highres':
            image_grid = process_highres_image(image, image_size=image_size, use_thumbnail=use_thumbnail, padding_value=padding_value)
        elif aspect_ratio == 'anyres':
            image_grid = process_anyres_image(image, image_size=image_size, use_thumbnail=use_thumbnail, padding_value=padding_value)
        elif aspect_ratio == 'adares':
            image_grid = process_adares_image(image, image_size=image_size, use_thumbnail=use_thumbnail)
        else:
            image_grid = [image]

        image_grid = [processor.preprocess(image_row, return_tensors='pt', num_images=len(images)) for image_row in image_grid]
        image_grids.append(image_grid)

    return image_grids


def frame_sample(duration, mode='uniform', num_frames=None, vid_fps=None, fps=None, must_sample_frames=None):
    mask_ids = []
    if mode == 'uniform':
        assert num_frames is not None, "Number of frames must be provided for uniform sampling."
        if duration <= num_frames:
            video_ids = np.arange(duration).astype(int)
            video_ids_list = video_ids.tolist()
            for msf in must_sample_frames:
                if msf not in video_ids_list:
                    video_ids_list.append(msf)
            video_ids_list.sort()
            for msf in must_sample_frames:
                mask_ids.append(video_ids_list.index(msf))
            return np.array(video_ids_list), mask_ids
        video_ids = np.linspace(0, duration-1, num_frames, dtype=int)
        video_ids_list = video_ids.tolist()
        if must_sample_frames is not None:
            for msf in must_sample_frames:
                if msf not in video_ids_list:
                    video_ids_list.append(msf)
            video_ids_list.sort()
            for msf in must_sample_frames:
                mask_ids.append(video_ids_list.index(msf))
        return np.array(video_ids_list), mask_ids
    elif mode == 'fps':
        assert vid_fps is not None, "FPS must be provided for FPS sampling."
        fps = fps if fps is not None else NUM_FRAMES_PER_SECOND
        segment_len = min(vid_fps // fps, duration)
        video_ids = np.arange(segment_len // 2, duration, segment_len, dtype=int)
        video_ids_list = video_ids.tolist()
        if must_sample_frames is not None:
            for msf in must_sample_frames:
                if msf not in video_ids_list:
                    video_ids_list.append(msf)
            video_ids_list.sort()
            for msf in must_sample_frames:
                mask_ids.append(video_ids_list.index(msf))
        return np.array(video_ids_list), mask_ids

    else:
        raise ImportError(f'Unsupported frame sampling mode: {mode}')


def load_video_from_ids(video_path, s=None, e=None, fps=None, max_frames=None, temporal_factor=1, must_sample_frames=None):
    if s is not None and e is not None:
        s = s if s >= 0. else 0.
        e = e if e >= 0. else 0.
        if s > e:
            s, e = e, s
        elif s == e:
            e = s + 1

    # 1. Loading Video
    if os.path.isdir(video_path):
        frame_files = sorted(os.listdir(video_path))

        vid_fps = 3
        num_frames_of_video = len(frame_files)
    elif video_path.endswith('.gif'):
        gif_reader = imageio.get_reader(video_path)

        vid_fps = 25
        num_frames_of_video = len(gif_reader)
    else:
        vreader = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        # vreader = VideoReader(video_path, ctx=cpu(0), num_threads=1)

        vid_fps = vreader.get_avg_fps()
        num_frames_of_video = len(vreader)

    # 2. Determine frame range & Calculate frame indices
    f_start = 0                       if s is None else max(int(s * vid_fps) - 1, 0)
    f_end   = num_frames_of_video - 1 if e is None else min(int(e * vid_fps) - 1, num_frames_of_video - 1)
    frame_indices = list(range(f_start, f_end + 1))

    duration = len(frame_indices)
    # 3. Sampling frame indices
    max_frames = max_frames if max_frames is not None else MAX_FRAMES
    if fps is not None and duration / vid_fps < max_frames:
        sampled_ids, mask_ids = frame_sample(duration, mode='fps', vid_fps=vid_fps, fps=fps, must_sample_frames=must_sample_frames)
        sampled_frame_indices = [frame_indices[i] for i in sampled_ids]
    else:
        sampled_ids, mask_ids = frame_sample(duration, mode='uniform', num_frames=max_frames, must_sample_frames=must_sample_frames)
        sampled_frame_indices = [frame_indices[i] for i in sampled_ids]

    # 4. Acquire frame data
    if os.path.isdir(video_path):
        frames = [cv2.cvtColor(cv2.imread(os.path.join(video_path, frame_files[frame_idx])), cv2.COLOR_BGR2RGB) for frame_idx in sampled_frame_indices]
    elif video_path.endswith('.gif'):
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB) for idx, frame in enumerate(gif_reader) if idx in sampled_frame_indices]
    else:
        frames = vreader.get_batch(sampled_frame_indices).asnumpy()

    # frames = frames.transpose(0, 3, 1, 2)
    timestamps = [x / vid_fps for x in sampled_frame_indices]

    if temporal_factor > 1:
        pad_length = temporal_factor - len(frames) % temporal_factor
        frames = np.concatenate([frames, frames[-1:].repeat(pad_length, axis=0)])
        [timestamps.append(timestamps[-1] + 1 / fps) for _ in range(pad_length)]

    # NOTE: pad the video with black frames
    # while num_frames is not None and len(video_data) < num_frames:
    #     video_data.append(Image.fromarray(np.zeros((*video_data[-1].size, 3), dtype=np.uint8)))

    return frames, timestamps, mask_ids


def load_video(
    video_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fps: Optional[float] = None,
    max_frames: Optional[float] = None,
    size: Optional[int] = None,
    size_divisible: int = 1,
    precise_time: bool = False,
    verbose: bool = False,
    temporal_factor: int = 1
):
    """
    Load and process a video file and return the frames and the timestamps of each frame.

    Args:
        video_path (str): Path to the video file.
        start_time (float, optional): Start time in seconds. Defaults to None.
        end_time (float, optional): End time in seconds. Defaults to None.
        fps (float, optional): Frames per second. Defaults to None.
        num_frames (float, optional): Number of frames to sample. Defaults to None.
        size (int, optional): Size of the shortest side. Defaults to None.
        size_divisible (int, optional): Size divisible by this number. Defaults to 1.
        precise_time (bool, optional): Whether to use precise time. Defaults to False.
        verbose (bool, optional): Print ffmpeg output. Defaults to False.

    Returns:
        frames (List[PIL.Image]): List of frames.
        timestamps (List[float]): List of timestamps.
    """
    if start_time is not None and end_time is not None and end_time - start_time < 1:
        return load_video_from_ids(video_path, start_time, end_time, fps=fps, max_frames=max_frames)
    if os.path.isdir(video_path):
        return load_video_from_ids(video_path, start_time, end_time, fps=fps, max_frames=max_frames)
    if video_path.endswith('.gif'):
        return load_video_from_ids(video_path, start_time, end_time, fps=fps, max_frames=max_frames)
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    w, h = int(video_stream['width']), int(video_stream['height'])

    kwargs, input_kwargs, output_kwargs = {}, {}, {}
    do_trim = start_time is not None or end_time is not None
    if start_time is not None:
        new_start_time = max(float(video_stream['start_time']), start_time)
        duration -= new_start_time - start_time
        start_time = new_start_time
    else:
        start_time = float(video_stream['start_time'])
    if end_time is not None:
        duration = min(duration, end_time - start_time)
    else:
        duration = duration
    if do_trim:
        kwargs = {'ss': start_time, 't': duration}
    if precise_time:
        output_kwargs.update(kwargs)
    else:
        input_kwargs.update(kwargs)

    if size is not None:
        scale_factor = size / min(w, h)
        new_w, new_h = round(w * scale_factor), round(h * scale_factor)
    else:
        new_w, new_h = w, h
    new_w = new_w // size_divisible * size_divisible
    new_h = new_h // size_divisible * size_divisible

    # NOTE: It may result in unexpected number of frames in ffmpeg
    # if calculate the fps directly according to max_frames
    # NOTE: the below lines may hurt the performance
    # if max_frames is not None and (fps is None or duration * fps > 2 * max_frames):
    #     fps = max_frames / duration * 2

    stream = ffmpeg.input(video_path, **input_kwargs)
    if fps is not None:
        stream = ffmpeg.filter(stream, "fps", fps=fps, round="down")
    if new_w != w or new_h != h:
        stream = ffmpeg.filter(stream, 'scale', new_w, new_h)
    stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt="rgb24", **output_kwargs)
    out, _ = ffmpeg.run(stream, capture_stdout=True, quiet=not verbose)

    frames = np.frombuffer(out, np.uint8).reshape([-1, new_h, new_w, 3]).transpose([0, 3, 1, 2])

    if fps is not None:
        timestamps = np.arange(start_time, start_time + duration + 1 / fps, 1 / fps)[:len(frames)]
    else:
        timestamps = np.linspace(start_time, start_time + duration, len(frames))

    max_frames = max_frames if max_frames is not None else MAX_FRAMES
    if max_frames is not None and len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = frames[indices]
        timestamps = [timestamps[i] for i in indices]

    if temporal_factor > 1:
        pad_length = temporal_factor - len(frames) % temporal_factor
        frames = np.concatenate([frames, frames[-1:].repeat(pad_length, axis=0)])
        [timestamps.append(timestamps[-1] + 1 / fps) for _ in range(pad_length)]

    frames = [frame for frame in frames]

    return frames, timestamps


def process_video(video_path, processor, s=None, e=None, aspect_ratio='pad', num_frames=None):
    fps = 1 if num_frames is None else None
    # FFmpeg
    frames, timestamps = load_video(video_path, s, e, fps=fps, max_frames=num_frames)
    # Decord
    # frames, timestamps = load_video_from_ids(video_path, s, e, fps=fps, max_frames=num_frames)

    assert len(frames) == len(timestamps), "Number of frames and timestamps must match."

    if aspect_ratio == 'pad':
        frames = [expand2square(f, tuple(int(x*255) for x in processor.image_mean)) for f in frames]

    if aspect_ratio == 'avt':
        frames = [processor.preprocess(frame, return_tensors='pt', image_num=len(frames)) for frame in frames]
        grid_frames = [frames]
    else:
        frames = processor.preprocess(frames, return_tensors='pt', image_num=len(frames))
        grid_frames = [[frames]]

    return grid_frames, timestamps


def tokenizer_multimodal_token(prompt, tokenizer, multimodal_token=DEFAULT_IMAGE_TOKEN, return_tensors=None):
    """Tokenize text and multimodal tag to input_ids.

    Args:
        prompt (str): Text prompt (w/ multimodal tag), e.g., '<video>\nDescribe the video.'
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer object.
        multimodal_token (int): Token index corresponding to the multimodal tag.
    """
    multimodal_token_index = MODAL_INDEX_MAP.get(multimodal_token, None)
    if multimodal_token_index is None:
        input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    else:
        prompt_chunks = [tokenizer(chunk, add_special_tokens=False).input_ids for idx, chunk in enumerate(prompt.split(multimodal_token))]

        input_ids = []
        for i in range(1, 2 * len(prompt_chunks)):
            if i % 2 == 1:
                input_ids.extend(prompt_chunks[i // 2])
            else:
                input_ids.append(multimodal_token_index)

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)

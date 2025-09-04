import cv2
import json
import os
import re
import copy
import argparse

from torchvision import io, transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mplc
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pycocotools import mask as maskUtils
from concurrent.futures import ThreadPoolExecutor, as_completed


def compress_data(data_dict, max_frame=100):
    if 'video' in data_dict and data_dict['video'] is not None:
        video_file = data_dict['video']
        timestamps = data_dict['timestamps']
        mask_ids = data_dict.get('mask_ids', [])

        num_frames = len(video_file)
        if num_frames > max_frame:
            last_frame_index = num_frames - 1
            reserved_indices = set(mask_ids + [last_frame_index])
            remaining_indices = [i for i in range(num_frames) if i not in reserved_indices]

            num_samples = max_frame - len(reserved_indices)

            sampled_non_fixed_indices = []
            if num_samples > 0:
                if len(remaining_indices) < num_samples:
                    sampled_non_fixed_indices = remaining_indices
                else:
                    interval = len(remaining_indices) / num_samples
                    sampled_non_fixed_indices = [remaining_indices[int(i * interval)] for i in range(num_samples)]


            final_indices = sorted(reserved_indices.union(sampled_non_fixed_indices))


            sampled_frames = [video_file[idx] for idx in final_indices]
            sampled_t = [timestamps[idx] for idx in final_indices] if timestamps else []

            new_mask_ids = [final_indices.index(idx) for idx in mask_ids if idx in final_indices]

            assert sampled_frames[-1] == video_file[-1]

            data_dict['video'] = sampled_frames
            data_dict['timestamps'] = sampled_t
            data_dict['mask_ids'] = new_mask_ids

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

def mask_to_box(mask):
    y_indices, x_indices = np.where(mask > 0)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None
    
    min_x, max_x = np.min(x_indices), np.max(x_indices)
    min_y, max_y = np.min(y_indices), np.max(y_indices)
    
    return (min_x, min_y, max_x, max_y)

def extract_frames(video_path, end_frame,frame_num=None,target_fps=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video: "+video_path)
        return []

    frame_count = 1
    imgs = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if target_fps:
        n_frames = fps // target_fps
    else:
        n_frames = (end_frame - 2) // (frame_num - 1)
    for frame_count in range(1,end_frame+1):
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_count-1) % n_frames == 0 or frame_count == end_frame:
            imgs.append([frame,(frame_count-1)/fps])
            if len(imgs) == frame_num and frame_count != end_frame:
                imgs.pop(-1)
            
    cap.release()
    return imgs,fps

def replace_with_objects(match, box):
    
    nums = tuple(map(int, match.groups()))
    if nums in box:
        current_index = box.index(nums)
        return f"<object {current_index}>"
    else:
        return match.group(0)
    
def draw_text_and_box(image, box_coords, texts, font_size=None, text_color='green', box_colors=['green'], text_alignment='center', rotation=0):
    """
    Draws a box and text on an image.

    Args:
        image (PIL.Image.Image): The image to draw on.
        box_coord (tuple): A tuple containing x0, y0, x1, y1 coordinates for the box.
        text (str): Text to draw.
        font_size (int): Font size for the text.
        text_color (str): Color of the text.
        box_color (str): Color of the box outline.
        text_alignment (str): Horizontal alignment of text ('left', 'center', 'right').
        rotation (float): Rotation angle for the text (in degrees).

    Returns:
        PIL.Image.Image: The image with the drawn box and text.
    """
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    height, width, _ = img_array.shape

    # Set up figure with the appropriate size and DPI
    fig = plt.figure(frameon=False)
    dpi = image.info.get('dpi', (72, 72))  # Default to 72 if DPI info is missing
    fig.set_dpi(dpi[0])
    fig.set_size_inches(width / dpi[0], height / dpi[0])

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    # Display the image
    ax.imshow(img_array, extent=(0, width, height, 0), interpolation='nearest')
    
    for box_coord, text, box_color in zip(box_coords, texts, box_colors):

        # Draw the rectangle (box)
        x0, y0, x1, y1 = box_coord
        width_box = x1 - x0
        height_box = y1 - y0
        if font_size is None:
            font_size = max(12, int(min(width, height) * 0.02))
        linewidth = font_size / 6 if font_size else 1
        rect = patches.Rectangle((x0, y0), width_box, height_box, linewidth=linewidth, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)

        if text is not None:
            # Compute the center position for the text
            text_position = ((x0 + x1) / 2, (y0 + y1) / 2)

            # Draw the text
            ax.text(
                text_position[0], text_position[1], text, fontsize=font_size, color=text_color,
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(facecolor='black', alpha=0.8, pad=0.7, edgecolor='none'), rotation=rotation
            )

    # Render the figure as an image
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    # Convert back to PIL image while preserving the original image size
    output_image = Image.fromarray(img_array)
    output_image = output_image.resize((width, height), Image.Resampling.LANCZOS)
    output_image.info['dpi'] = dpi  # Ensure DPI info is kept

    return output_image


def generate_dataset(idx, data, image_folder, data_folder):
    try:
        # pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
        pattern = r'(<object\d+>) <region>'
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink']

        compress_data(data, max_frame=30)
        video_file_raw = data['video']
        conversation = copy.deepcopy(data["conversations"])
        
        video_root = data['video_id']
        video_file = [os.path.join(video_root, vf) for vf in video_file_raw]

        images = []
        for vf in video_file:
            image = Image.open(os.path.join(data_folder,vf))
            images.append(image)

        for conv in conversation:
            conv['value'] = conv['value'].replace('<video>', '')
            if conv["from"] == "human":
                question = conv["value"]
            else:
                answer = conv["value"]

        masks = data['masks']
        mask_ids = data['mask_ids']

        new_video_paths = []
        new_folder = os.path.join(image_folder, f'{idx:05d}')
        os.makedirs(new_folder, exist_ok=True)
        if masks is not None:
            matches = re.findall(pattern, question)
            if len(matches) > 0:
                assert len(matches) == len(masks)
            else:
                matches = ['<object0>']
                question = question.replace('<region>', '<object0>')
            
            i = 0
            boxes = {}
            texts = {}
            box_colors = {}
            for midx, ann in enumerate(masks):
                box_text = matches[midx]
                box_color = colors[midx]
                for k in ann.keys():
                    mask = annToMask(ann[k], None, None)
                    box = mask_to_box(mask)
                    if mask_ids[i] not in boxes:
                        boxes[mask_ids[i]] = []
                        texts[mask_ids[i]] = []
                        box_colors[mask_ids[i]] = []
                    boxes[mask_ids[i]].append(box)
                    texts[mask_ids[i]].append(box_text)
                    box_colors[mask_ids[i]].append(box_color)
                    i += 1
            
            
            
            for frame_idx, image in enumerate(images):
                frame_path = os.path.join(new_folder, os.path.basename(video_file[frame_idx]))
                if frame_idx in boxes:
                    if image.size[0]>600:
                        font_size=15
                    else:
                        font_size=8
                    image = draw_text_and_box(image, boxes[frame_idx], texts[frame_idx], font_size=font_size, text_color='white', box_colors=box_colors[frame_idx])
                
                image.save(frame_path)
                new_video_paths.append(frame_path)

            question = question.replace('<region>', '')
        
        else:
            for frame_idx, image in enumerate(images):
                frame_path = os.path.join(new_folder, os.path.basename(video_file[frame_idx]))
                image.save(frame_path)
                new_video_paths.append(frame_path)


        meta_info = {
            "idx": idx,
            "video" : new_video_paths,
            "question" : question,
            "answers" : answer,
            'type': data.get("type"),
            'class_name': data.get("class", None)
        }

        return meta_info
    except Exception as e:
        print(f'{idx} Error: {e}')
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset with bounding boxes and texts.")
    parser.add_argument("--data_json", type=str, help="Path to the input JSON file containing data.")
    parser.add_argument("--data_folder", type=str, help="Root folder where the original videos/images are stored.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the output images and metadata.")
    args = parser.parse_args()

    datas_json = args.data_json
    output_dir = args.output_dir
    data_folder = args.data_folder

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_folder = os.path.join(output_dir, 'images')

    with open(datas_json,"r") as f:
        datas = json.load(f)

    meta_path = os.path.join(output_dir,"meta_infos.json")

    meta_infos = []

    with ThreadPoolExecutor(max_workers=8) as executor:  # 根据机器资源合理配置max_workers
        futures = {
            executor.submit(generate_dataset, idx, data, image_folder, data_folder): idx for idx, data in enumerate(datas)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating datasets"):
            meta_info = future.result()
            if meta_info is not None:
                meta_infos.append(meta_info)


    with open(meta_path, "w") as f:
        json.dump(meta_infos, f, indent=4)
    

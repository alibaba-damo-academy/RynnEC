import argparse
import sys
sys.path.append('./')
import re

import cv2
import torch
import gradio as gr
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from transformers import SamModel, SamProcessor
import json
import numpy as np
from PIL import Image
import os
import math
from tqdm import tqdm
from torchvision.transforms import v2

from rynnec import disable_torch_init, model_init, mm_infer, mm_infer_segmentation
from rynnec.mm_utils import annToMask, load_video, load_images

from visualize_mask import colorize_masks


color_rgb = (1.0, 1.0, 1.0)
color_rgbs = [
    (1.0, 1.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 1.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
]


def extract_first_frame_from_video(video):
    cap = cv2.VideoCapture(video)
    success, frame = cap.read()
    cap.release()
    if success:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None


def extract_points_from_mask(mask_pil):
    mask = np.asarray(mask_pil)[..., 0]
    coords = np.nonzero(mask)
    coords = np.stack((coords[1], coords[0]), axis=1)

    return coords

def add_contour(img, mask, color=(1., 1., 1.)):
    img = img.copy()

    mask = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness=8)

    return img


def load_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise gr.Error("Could not read the video file.")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)  
    return image


def clear_masks():
    return [], [], [], []

def clear_all():
    return [], [], [], [], None, "", ""


def apply_sam(image, input_points):
    inputs = sam_processor(image, input_points=input_points, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0][0]
    scores = outputs.iou_scores[0, 0]

    mask_selection_index = scores.argmax()

    mask_np = masks[mask_selection_index].numpy()

    return mask_np


def run(mode, images, timestamps, masks, mask_ids, instruction, mask_output_video):
    if mode == "QA":
        response = run_text_inference(images, timestamps, masks, mask_ids, instruction)
    else:
        response, mask_output_video = run_seg_inference(images, timestamps, instruction)
    return response, mask_output_video


def run_text_inference(images, timestamps, masks, mask_ids, instruction):
    masks = torch.from_numpy(np.stack(masks, axis=0))

    if "<video>" not in instruction:
        instruction = "<video>\n" + instruction

    if len(masks) >= 2:
        obj_str = f"<video>\nThere are {len(masks)} objects in the video: " + ", ".join([f"<object{i}> [<REGION>]" for i in range(len(masks))])
        instruction = instruction.replace("<video>\n", obj_str)
    else:
        instruction = instruction.replace("<object0>", '[<REGION>]')

    output = mm_infer(
        (images, timestamps),
        processor,
        instruction,
        model=model,
        tokenizer=processor.tokenizer,
        do_sample=False,
        modal='video',
        masks=masks.cuda() if masks is not None else None,
        mask_ids=mask_ids
    )

    return output


def run_seg_inference(images, timestamps, instruction):
    output, masks = mm_infer_segmentation(
        (images, timestamps),
        processor,
        instruction,
        model=model,
        tokenizer=processor.tokenizer,
        do_sample=False,
        modal='video',
    )

    w, h = images[0].size
    masks = v2.Resize([h, w])(masks).cpu().numpy()

    mask_list_video = []

    images = [np.array(image) for image in images]
    masks = [mask[0] for mask in masks]
    show_images, _ = colorize_masks(images, masks)
    for i, image in enumerate(show_images):
        if masks[i].sum() > 1000:
            mask_list_video.append((Image.fromarray(image), f"Frame {i}"))        

    return output, mask_list_video


def generate_masks_video(image, mask_list_video, mask_raw_list_video, mask_ids, frame_idx):
    image['image'] = image['background'].convert('RGB')
    # del image['background'], image['composite']
    assert len(image['layers']) == 1, f"Expected 1 layer, got {len(image['layers'])}"

    mask = Image.fromarray((np.asarray(image['layers'][0])[..., 3] > 0).astype(np.uint8) * 255).convert('RGB')
    points = extract_points_from_mask(mask)
    np.random.seed(0)
    if points.shape[0] == 0:
        raise gr.Error("No points selected")

    points_selected_indices = np.random.choice(points.shape[0], size=min(points.shape[0], 8), replace=False)
    points = points[points_selected_indices]
    coords = [points.tolist()]
    mask_np = apply_sam(image['image'], coords)

    mask_raw_list_video.append(mask_np)
    mask_image = Image.fromarray((mask_np[:,:,np.newaxis] * np.array(image['image'])).astype(np.uint8))
    
    mask_list_video.append((mask_image, f"<object{len(mask_list_video)}>"))
    # Return a list containing the mask image.
    image['layers'] = []
    image['composite'] = image['background']
    mask_ids.append(frame_idx)
    return mask_list_video, image, mask_list_video, mask_raw_list_video, mask_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoRefer gradio demo")
    parser.add_argument("--model-path", type=str, default="Alibaba-DAMO-Academy/RynnEC-2B", help="Path to the model checkpoint")
    parser.add_argument("--port", type=int, default=4001)

    args_cli = parser.parse_args()

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="amber")) as demo:

        mask_list = gr.State([])  
        mask_raw_list = gr.State([])  
        mask_list_video = gr.State([])  
        mask_raw_list_video = gr.State([])  


        HEADER = ("""
            <div>
                <h1>RynnEC Demo</h1>
                <h5 style="margin: 0;">Feel free to click on anything that grabs your interest!</h5>
                <h5 style="margin: 0;">If this demo please you, please give us a star ⭐ on Github or 💖 on this space.</h5>
            </div>
            </div>
            <div style="display: flex; justify-content: left; margin-top: 10px;">
            <a href="https://arxiv.org/pdf/2501.00599"><img src="https://img.shields.io/badge/Arxiv-2501.00599-ECA8A7" style="margin-right: 5px;"></a>
            <a href="https://github.com/DAMO-NLP-SG/VideoRefer"><img src='https://img.shields.io/badge/Github-VideoRefer-F7C97E' style="margin-right: 5px;"></a>
            <a href="https://github.com/DAMO-NLP-SG/VideoLLaMA3"><img src='https://img.shields.io/badge/Github-VideoLLaMA3-9DC3E6' style="margin-right: 5px;"></a>
            </div>
            """)


        image_tips = """
                ### 💡 Tips:

                🧸 Upload an image, and you can use the drawing tool✍️ to highlight the areas you're interested in.
            
                🔖 For single-object caption mode, simply select the area and click the 'Generate Caption' button to receive a caption for the object.
                
                🔔 In QA mode, you can generate multiple masks by clicking the 'Generate Mask' button multiple times. Afterward, use the corresponding object id to ask questions.
                
                📌 Click the button 'Clear Masks' to clear the current generated masks.
                
                """
        
        video_tips = """
                ### 💡 Tips:
                🧸 Upload an video, and you can use the drawing tool✍️ to highlight the areas you're interested in the first frame.
                
                🔔 In QA mode, you can generate multiple masks by clicking the 'Generate Mask' button multiple times. Afterward, use the corresponding object id to ask questions.
                
                📌 Click the button 'Clear Masks' to clear the current generated masks.
                
                """


        with gr.TabItem("Video"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Video", interactive=True)
                    frame_idx = gr.Slider(minimum=0, maximum=0, value=0, step=1, label="Select Frame", interactive=False)
                    selected_frame = gr.ImageEditor(
                        label="Annotate Frame",
                        type="pil", 
                        sources=[], 
                        interactive=True,
                    )
                    generate_mask_btn_video = gr.Button("1️⃣ Generate Mask", visible=True, variant="primary")
                    gr.Examples([f"./demo/videos/{i+1}.mp4" for i in range(4)], inputs=video_input, label="Examples")

                with gr.Column():
                    mode_video = gr.Radio(label="Mode", choices=["QA", "Seg"], value="QA")
                    mask_output_video = gr.Gallery(label="Referred Masks", object_fit='scale-down')

                    query_video = gr.Textbox(label="Question", value="Please describe <object0>.", interactive=True, visible=True)
                    response_video = gr.Textbox(label="Answer", interactive=False)

                    submit_btn_video = gr.Button("Generate Caption", variant="primary", visible=False)
                    submit_btn_video1 = gr.Button("2️⃣ Generate Answer", variant="primary", visible=True)
                    description_video = gr.Textbox(label="Output", visible=False)
                    
                    clear_masks_btn_video = gr.Button("Clear Masks", variant="secondary")

            gr.Markdown(video_tips)

            frames = gr.State(value=[])
            timestamps = gr.State(value=[])
            mask_ids = gr.State(value=[])

        def on_video_upload(video_path):
            frames, timestamps = load_video(video_path, fps=1, max_frames=128)
            frames = [Image.fromarray(x.transpose(1, 2, 0)) for x in frames]
            return frames, timestamps, frames[0], gr.update(value=0, maximum=len(frames) - 1, interactive=True)

        def on_frame_idx_change(frame_idx, frames):
            return frames[frame_idx]

        def to_seg_mode():
            return (
                *[gr.update(visible=False) for _ in range(4)],
                []
            )

        def to_qa_mode():
            return (
                *[gr.update(visible=True) for _ in range(4)],
                []
            )

        def on_mode_change(mode):
            if mode == "QA":
                return to_qa_mode()
            return to_seg_mode()

        mode_video.change(on_mode_change, inputs=[mode_video], outputs=[frame_idx, selected_frame, generate_mask_btn_video, response_video, mask_output_video])
        video_input.change(on_video_upload, inputs=[video_input], outputs=[frames, timestamps, selected_frame, frame_idx])
        frame_idx.change(on_frame_idx_change, inputs=[frame_idx, frames], outputs=[selected_frame])

        generate_mask_btn_video.click(
            fn=generate_masks_video,
            inputs=[selected_frame, mask_list_video, mask_raw_list_video, mask_ids, frame_idx],
            outputs=[mask_output_video, selected_frame, mask_list_video, mask_raw_list_video, mask_ids]
        )

        submit_btn_video1.click(
            fn=run,
            inputs=[mode_video, frames, timestamps, mask_raw_list_video, mask_ids, query_video, mask_output_video],
            outputs=[response_video, mask_output_video],
            api_name="describe_video"
        )

        video_input.clear(
            fn=clear_all,
            outputs=[mask_output_video, mask_list_video, mask_raw_list_video, mask_ids, selected_frame, query_video, response_video]
        )

        clear_masks_btn_video.click(
            fn=clear_masks,
            outputs=[mask_output_video, mask_list_video, mask_raw_list_video, mask_ids]
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    # sam_model = sam_processor = None
    disable_torch_init()
    model, processor = model_init(args_cli.model_path)
    # model = processor = None

    # demo.launch()
    demo.launch(
        share=False,
    )

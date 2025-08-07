import cv2
import json
import os
import re
import copy
import glob
import random

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
import colorsys
import argparse

def get_hsv_palette(n_colors):
    hues = np.linspace(0, 1, int(n_colors) + 1)[1:-1] 
    s = 0.8    
    v = 0.9   
    palette = [(0.0, 0.0, 0.0)] + [
        colorsys.hsv_to_rgb(h_i, s, v) for h_i in hues
    ]
    return (255 * np.asarray(palette)).astype("uint8")


def colorize_masks(images, index_masks, fac: float = 0.8, draw_contour=True, edge_thickness=20):
    max_idx = max([m.max() for m in index_masks])
    palette = get_hsv_palette(max_idx + 1)
    color_masks = []
    out_frames = []
    for img, mask in tqdm(zip(images, index_masks), desc='Visualize masks ...'):
        clr_mask = palette[mask.astype("int")]
        blended_img = img

        blended_img = compose_img_mask(blended_img, clr_mask, fac)

        if draw_contour:
            blended_img = draw_contours_on_image(blended_img, mask, clr_mask,
                                                 brightness_factor=1.8,
                                                 alpha=0.6,
                                                 thickness=edge_thickness)
        out_frames.append(blended_img)

    return out_frames, color_masks


def compose_img_mask(img, color_mask, fac: float = 0.5):
    mask_region = (color_mask.sum(axis=-1) > 0)[..., None]
    out_f = img.copy() / 255
    out_f[mask_region[:, :, 0]] = fac * img[mask_region[:, :, 0]] / 255 + (1 - fac) * color_mask[mask_region[:, :, 0]] / 255
    out_u = (255 * out_f).astype("uint8")
    return out_u

def draw_contours_on_image(img, index_mask, color_mask, brightness_factor=1.6, alpha=0.5, thickness=2, ignore_index=0):
    img = img.astype("float32")
    overlay = img.copy()

    unique_indices = np.unique(index_mask)
    if ignore_index is not None:
        unique_indices = [idx for idx in unique_indices if idx != ignore_index]

    for i in unique_indices:
        bin_mask = (index_mask == i).astype("uint8") * 255
        if bin_mask.sum() == 0:
            continue

        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color = color_mask[index_mask == i][0].astype("float32")
        bright_color = np.clip(color * brightness_factor, 0, 255).tolist()

        cv2.drawContours(overlay, contours, -1, bright_color, thickness)

    blended = (1 - alpha) * img + alpha * overlay
    return np.clip(blended, 0, 255).astype("uint8")



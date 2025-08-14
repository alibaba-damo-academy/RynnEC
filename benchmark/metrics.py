import torch
from openai import OpenAI, AzureOpenAI, APIError
import re
import math
import numpy as np
import cv2
from typing import List

client = OpenAI(
    api_key=API_KEY, # use your api
    base_url=URL, # use your url
)


def calculate_iou(pred_mask, gt_mask):
    pred_mask = pred_mask.bool().reshape(-1)
    gt_mask = gt_mask.bool().reshape(-1)

    intersection = torch.logical_and(pred_mask, gt_mask).sum()
    union = torch.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union

    return iou


TYPE_LIST = {
    "Object Properties Cognition": [
        "category",
        "color",
        "material",
        "shape",
        "state",
        "position",
        "function",
        "surface detail",
        "size",
        "counting"
    ],
    "Referring Object Segmentation": [
        "Direct Referring",
        "Situational Referring"
    ],
    "Self-Centered": {
        "Trajectory Review": ["camera_rotation", "camerall_distance"],
        "Ego Direction": ["relative_direction_to_camera", "distance_to_camera", "direction_to_camera", "direction_to_camera_3", "relative_direction_to_camera_2"],
        "Ego Distance": ["distance_to_camera_choice_3", "distance_to_camera_2", "relative_distance_to_object_choice_2"],
        "Movement Imagery": [
            "future_direction_to_camera",
            "future_direction_to_camera_rotate"],
        "Spatial Imagery":[
            "future_direction_object_3"
        ]
    },
    "World-Centered": {
        "Size": [
            "height_data",
            "tall_choice_3",
            "small_predicate_3",
            "tall_predicate",
            "tall_choice",
            "short_predicate",
            "short_choice",
            "big_predicate",
            "small_predicate",
        ],
        "Object height": ["height_from_ground"],
        "Object Distance": [
            "distance_compare",
            "center_distance"
        ],
        "Absolute Position": [
            "above_predicate",
            "above_choice",
            "below_predicate",
            "below_choice"],

        "Rel Position": ["between",
            "directly_above"
        ]
    }
}

SCORE_TYPE = {
        "camera_rotation": "numerical", 
        "camerall_distance": "numerical",
        "relative_direction_to_camera": "gpt", 
        "distance_to_camera": "numerical", 
        "direction_to_camera": "gpt", 
        "direction_to_camera_3": "gpt", 
        "distance_to_camera_choice_3": "gpt", 
        "relative_direction_to_camera_2": "gpt", 
        "distance_to_camera_2": "numerical", 
        "relative_distance_to_object_choice_2": "gpt",
        "future_direction_to_camera": "gpt", 
        "future_direction_to_camera_rotate": "numerical", 
        "future_direction_object_3": "gpt",
        "height_data": "numerical", 
        "tall_choice_3": "gpt", 
        "small_predicate_3": "gpt", 
        "tall_predicate": "gpt", 
        "tall_choice": "gpt", 
        "short_predicate": "gpt", 
        "short_choice": "gpt", 
        "big_predicate": "gpt", 
        "small_predicate": "gpt", 
        "height_from_ground": "numerical",
        "distance_compare": "gpt", 
        "center_distance": "numerical",
        "between": "gpt", 
        "above_predicate": "gpt", 
        "above_choice": "gpt", 
        "below_predicate": "gpt",
        "below_choice": "gpt", 
        "directly_above": "gpt", 
        "universal": "gpt_multi_granularity",
        "counting": "gpt",
        "category": "gpt_multi_granularity",
        "color": "gpt_multi_granularity",
        "material": "gpt_multi_granularity",
        "shape": "gpt_multi_granularity",
        "state": "gpt_multi_granularity",
        "position": "gpt_multi_granularity",
        "function": "gpt_multi_granularity",
        "surface detail": "gpt_multi_granularity",
        "size": "gpt_multi_granularity",
}

UNIT_LIST = ['centimeters', 'meters', 'feet', 'inches', 'degrees', 'o\'clock']

UNIT_EXCHANGE = {
    'length': {'centimeters': 100., "meters": 1., "inches": 39.3701, "feet": 3.28084},
}

def flatten_structure(d):
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            flat_values = []
            for subkey, subvalue in value.items():
                flat_values.extend(subvalue)
                result[subkey] = subvalue
            result[key] = flat_values
        else:
            result[key] = value
    return result


def clean_text(text):
    text = text.replace("<video>", "")
    text = text.replace("<REGION>", "")

    text = re.sub(r'[\[\]]', '', text)
    text = text.strip()
    return text

def make_prompt(question, label_answer, pred_answer):
    # question = clean_text(question)
    prompt = f'###Question: {question} ###Label Answer: {label_answer} ###Predicted Answer: {pred_answer}'
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Please judge whether the predicted answer is correct or not according to the given question and labeled answer. You need to consider the answer from two perspectives: accuracy and completeness. Output ###Judge: True only when the predicted answer is accurate and complete; otherwise, output ###Judge: False"},
        {"role": "user", "content": '###Question: I need you to clean the dust on the red-marked line in this area. How many tables and chairs should you move before cleaning? ###Label Answer: 0 tables and 1 chair. ###Predicted Answer: One chair'},
        {"role": "assistant", "content": "###Judge: False"},
        {"role": "user", "content": '###Question: If you are now standing outside the restroom stall door away from the sink. Which edge will the door rotate around if you move forward? ###Label Answer: Using the left side edge as the axis. ###Predicted Answer: The door will rotate around the left edge'},
        {"role": "assistant", "content": "###Judge: True"},
        {"role": "user", "content": '###Question: If you start from the position of the backpack next to the sofa and keep moving towards the direction of the bicycle. What would happen if you stop immediately after hitting the bicycle? What if you don\'t stop immediately but continue moving forward? ###Label Answer: If you stop immediately after hitting the bicycle, the bicycle will sway slightly but won\'t fall over. If you continue moving forward, the bicycle will tilt towards the window until it leans against it. ###Predicted Answer: If you stop immediately after hitting the bicycle, you would likely stabilize the bicycle and yourself. If you don\'t stop immediately but continue moving forward, you might knock over the bicycle and potentially cause damage or injury'},
        {"role": "assistant", "content": "###Judge: False"},
        {"role": "user", "content": '###Question: I took some documents from the drawer in this room earlier, please help me check if there are any drawers that haven\'t been pushed in completely. ###Label Answer: No. ###Predicted Answer: All drawers appear to be closed properly'},
        {"role": "assistant", "content": "###Judge: True"},
        {"role": "user", "content": prompt},
        ]
    return messages

def make_prompt_multi_granularity(question, label_answer, pred_answer):
    # question = clean_text(question)
    prompt = f'###Question: {question} ###5 Score Answer: {label_answer} ###Predicted Answer: {pred_answer}'
    
    question1 = 'If you pick up the footstool between the sofa and the TV and flip it 180° upside down, which objects in the scene will be affected and how?'
    label_answer1 = 'Three handles and one remote control will fall onto the carpet.'
    pred_answer1 = 'remote control will fall onto the carpet'
    score1 = '3'


    question2 = 'If you pick up the footstool between the sofa and the TV and flip it 180° upside down, which objects in the scene will be affected and how?'
    label_answer2 = 'Three handles and one remote control will fall onto the carpet.'
    pred_answer2 = 'handles'
    score2 = '2'


    question3 = 'If you pick up the footstool between the sofa and the TV and flip it 180° upside down, which objects in the scene will be affected and how?'
    label_answer3 = 'Three handles and one remote control will fall onto the carpet.'
    pred_answer3 = 'handles and control will fall '
    score3 = '4'


    question4 =  'Which area in this room is better for studying? Why?'
    label_answer4 = 'The U-shaped table area, which has matching chairs and good lighting'
    pred_answer4 = 'may be table area'
    score4 = '1'


    question5 =  'Which area in this room is better for studying? Why?'
    label_answer5 =   'The U-shaped table area, which has matching chairs and good lighting'
    pred_answer5 =  'Over there by the sofa.'
    score5 = '0'


    question6 =  'Which area in this room is better for studying? Why?'
    label_answer6 =   'The U-shaped table area, which has matching chairs and good lighting'
    pred_answer6 =  'may be table area and chairs'
    score6 = '4'


    messages=[
        {"role": "system", "content": "You are a helpful assistant. Please score the predicted answer according to the given question and huamn labeled the 5-score answer, and the 3-score answer. 0 score represents completely wrong, 5 scores represents completely correct, and 3 scores represents partially correct. Please refer to them to score the predicted answer: [0, 1, 2, 3, 4, 5]. You need to consider the answer from two perspectives: accuracy and completeness. Output ###Judge:"},
        {"role": "user", "content": f"###Question: {question1} ###5 Score Answer: {label_answer1}. ###Predicted Answer: {pred_answer1}"},
        {"role": "assistant", "content": f"###Judge: {score1}"},
        {"role": "user", "content": f"###Question: {question2} ###5 Score Answer: {label_answer2}. ###Predicted Answer: {pred_answer2}"},
        {"role": "assistant", "content": f"###Judge: {score2}"},
        {"role": "user", "content": f"###Question: {question3} ###5 Score Answer: {label_answer3}. ###Predicted Answer: {pred_answer3}"},
        {"role": "assistant", "content": f"###Judge: {score3}"},
        {"role": "user", "content": f"###Question: {question4} ###5 Score Answer: {label_answer4}. ###Predicted Answer: {pred_answer4}"},
        {"role": "assistant", "content": f"###Judge: {score4}"},
        {"role": "user", "content": f"###Question: {question5} ###5 Score Answer: {label_answer5}. ###Predicted Answer: {pred_answer5}"},
        {"role": "assistant", "content": f"###Judge: {score5}"},
        {"role": "user", "content": f"###Question: {question6} ###5 Score Answer: {label_answer6}. ###Predicted Answer: {pred_answer6}"},
        {"role": "assistant", "content": f"###Judge: {score6}"},
        {"role": "user", "content": prompt},
        ]

    return messages

def score_by_gpt(item, score_type="binary"):

    prompt_maker = {
        "binary": make_prompt,
        "multi": make_prompt_multi_granularity
    }

    question = item['Question']
    label_answer = item['Answer']
    pred_answer =  item['pred']
    question = clean_text(question)
    messages = prompt_maker[score_type](question, label_answer, pred_answer) 

    try:
        completion = client.chat.completions.create(
            model='gpt-4o-0806',
            messages=messages,
            max_tokens=100,
        )
        response_raw = completion.choices[0].message.content 
        response = response_raw.split("###Judge:")[1].strip()

        try:
            if score_type == "binary":
                score = 1 if response == 'True' else 0
            else:
                match = re.match(r'(\d+)(.*)', response)
                response = match.group(1)
                score = float(int(response) / 5)

            return score, True
        except:
            print(f"Model output: {response_raw}\n")
            # print(f"Prompt messages sent to model: {messages}")
            return 0., False
    except Exception as e:
        print(f"Error: {e}")
        # print(f"Prompt messages sent to model: {messages}")
        return score_by_gpt(item, score_type)
    
def extract_number_and_unit(text):
    units_regex = '|'.join(UNIT_LIST)
    
    match = re.search(r'(\d+(\.\d+)?)\s*(' + units_regex + r')', text, re.IGNORECASE)
    if match:
        return float(match.group(1)), match.group(3)
    else:
        return None, None
    
def calculate_MRA(number_result, number_gt):
    seta_group = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    metric = 0
    for seta in seta_group:
        if abs(number_result - number_gt) / number_gt  < 1 - seta:
            metric += 0.1
    return metric

def calculate_MRA_degree(number_result, number_gt, ref = 90):
    seta_group = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    metric = 0
    for seta in seta_group:
        if abs(number_result - number_gt) / ref  < 1 - seta:
            metric += 0.1
    return metric

def numerical_score(item):
    label_answer = item['Answer']
    pred_answer =  item['pred']

    number_gt, unit_gt = extract_number_and_unit(label_answer)
    number_pred, unit_pred = extract_number_and_unit(pred_answer)

    if number_pred is None:
        return 0.
    
    if unit_gt in UNIT_EXCHANGE["length"]:
        if unit_pred not in UNIT_EXCHANGE["length"]:
            score = 0.
        else:
            number_gt = number_gt / UNIT_EXCHANGE["length"][unit_gt]
            number_pred = number_pred / UNIT_EXCHANGE["length"][unit_pred]
            score = calculate_MRA(number_pred, number_gt)
    
    elif "o\'clock" in unit_gt:
        score = max(calculate_MRA_degree(number_pred, number_gt, ref=3), calculate_MRA_degree(number_pred-12, number_gt, ref=3))
    elif "degree" in unit_gt:
        score = calculate_MRA_degree(number_pred, number_gt, ref=90)
    
    else:
        score = calculate_MRA(number_pred, number_gt)
        
    return score

def calculate_score(d):
    if isinstance(d['type'], list):
        type_ = d["type"][0]
    else:
        type_ = d["type"]
    score_type = SCORE_TYPE[type_.lower()]
    if score_type == "gpt":
        score, success = score_by_gpt(d, "binary")
    elif score_type == "gpt_multi_granularity":
        score, success = score_by_gpt(d, "multi")
    elif score_type == "numerical":
        score = numerical_score(d)
        success = True

      
    return score
    



def get_r2vos_accuracy(gt_masks: List[np.ndarray], pred_masks: List[np.ndarray], ):
    """ Compute per-pixel accuracy.
    Args:
        gt_masks: List[np.ndarray], shape: (n_frames, h, w), dtype: np.uint8
        pred_masks: List[np.ndarray], shape: (n_frames, h, w), dtype: np.uint8
    Return:
        accs: np.ndarray, shape: (n_frames,), dtype: np.float32
    """
    assert len(gt_masks) == len(pred_masks), "The number of frames in gt_masks and pred_masks should be the same"
    n_frames = len(gt_masks)
    accs = []
    for gt_mask, pred_mask in zip(gt_masks, pred_masks):
        assert gt_mask.shape == pred_mask.shape, "The shape of gt_mask and pred_mask should be the same"
        acc = np.mean(gt_mask == pred_mask)
        accs.append(acc)
    return np.array(accs)

def get_r2vos_robustness(gt_masks: List[np.ndarray], pred_masks: List[np.ndarray], foreground_masks: List[np.ndarray], ):
    """
    Args:
        gt_masks: List[np.ndarray], shape: (n_frames, h, w), dtype: np.uint8
        pred_masks: List[np.ndarray], shape: (n_frames, h, w), dtype: np.uint8
        foreground_masks: List[np.ndarray], shape: (n_frames, h, w), dtype: np.uint8
    Return:
        robustness: np.ndarray, shape: (n_frames,), dtype: np.float32
    """
    assert len(gt_masks) == len(pred_masks) == len(foreground_masks), "The number of frames in gt_masks, pred_masks and foreground_masks should be the same"
    n_frames = len(gt_masks)
    robustness = []
    for gt_mask, pred_mask, foreground_mask in zip(gt_masks, pred_masks, foreground_masks):
        assert gt_mask.shape == pred_mask.shape == foreground_mask.shape, "The shape of gt_mask, pred_mask and foreground_mask should be the same"
        neg = ((1 - gt_mask) * pred_mask).sum()
        pos = foreground_mask.sum()
        robust = max(1 - neg / (pos + 1e-6), 0.0)
        robustness.append(robust)
    return np.array(robustness)

def db_eval_iou(annotation, segmentation, void_pixels=None):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels

    Return:
        jaccard (float): region similarity
    """
    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, \
            f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    # Intersection between all sets
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j


def db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008):
    assert annotation.shape == segmentation.shape, f'ann shape: {annotation.shape}, seg shape: {segmentation.shape}'
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :, ]
            f_res[frame_id] = f_measure(segmentation[frame_id, :, :, ], annotation[frame_id, :, :], void_pixels_frame, bound_th=bound_th)
    elif annotation.ndim == 2:
        f_res = f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
    else:
        raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
    return f_res


def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels

    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(bool)

    bound_pix = bound_th if bound_th >= 1 else np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    from skimage.morphology import disk

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap

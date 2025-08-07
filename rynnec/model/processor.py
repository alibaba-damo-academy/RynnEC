# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for VideoLLaMA3.
"""
from abc import ABCMeta, abstractmethod
import copy
import warnings
from collections import defaultdict
from typing import List, Union, Dict, Optional, Any

import json
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from rynnec.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from rynnec.mm_utils import load_video, load_images
from rynnec.model.videollama3_encoder.image_processing_videollama3 import is_valid_image, is_valid_video


class Videollama3ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class Videollama3BaseProcessor(ProcessorMixin, metaclass=ABCMeta):
    r"""
    Modified from Qwen2VLProcessor
    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "image_merge_size", "video_merge_size", "fps", "max_frames"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = None
    chat_template = None

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_merge_size: int = 1,
        video_merge_size: int = 2,
        fps=1,
        max_frames=180,
        **kwargs
    ):
        if chat_template is not None:
            self.chat_template = chat_template

        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_merge_size = image_merge_size
        self.video_merge_size = video_merge_size
        self.fps = fps
        self.max_frames = max_frames

        if self.chat_template is not None:
            self.tokenizer.chat_template = self.chat_template

        self.image_token = DEFAULT_IMAGE_TOKEN
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"
        self.tokenizer.add_tokens([self.image_token], special_tokens=True)
        self.tokenizer.add_tokens([self.think_start_token, self.think_end_token], special_tokens=False)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.think_start_token_id = self.tokenizer.convert_tokens_to_ids(self.think_start_token)
        self.think_end_token_id = self.tokenizer.convert_tokens_to_ids(self.think_end_token)
        self.newline_token_id = self.tokenizer.encode("\n")[0]

    def load_video(self, *args, **kwargs):
        return load_video(*args, **kwargs)

    def load_images(self, *args, **kwargs):
        return load_images(*args, **kwargs)
    
    def _get_downsampled_grid_sizes(self, image_inputs: Dict[str, Any]):
        grid_sizes = []
        for grid_size, merge_size in zip(image_inputs.get("grid_sizes", []), image_inputs.get("merge_sizes", [])):
            if not torch.all(grid_size[1:] % merge_size == 0):
                warnings.warn(f"Grid size {grid_size} is not divisible by merge size. Some undesired errors may occur.")
            if grid_size[0] == 1:
                grid_sizes.append(grid_size[1:] / merge_size)
            elif grid_size[0] > 1:
                grid_sizes.extend([grid_size[1:] / merge_size] * grid_size[0])
        return grid_sizes

    def _get_visual_seq_len(self, grid_size: torch.Tensor):
        num_tokens = int(grid_size.prod().item())
        return num_tokens

    @abstractmethod
    def _process_text_with_label(
        self,
        text: List[Dict],
        grid_sizes: torch.Tensor = None,
        **kwargs,
    ):
        return {}

    def _process_text_without_label(
        self,
        text: Union[List[str], List[Dict]],
        grid_sizes: torch.Tensor = None,
        **kwargs,
    ):
        if isinstance(text, (list, tuple)) and isinstance(text[0], dict):
            warnings.warn("Input text is a list of messages. Automatically convert it to a string with 'apply_chat_template' with generation prompt.")
            text = self.apply_chat_template(text, tokenize=False, add_generation_prompt=True)

        if len(grid_sizes) > 0:
            image_idx = 0
            while self.image_token in text:
                thw = grid_sizes[image_idx]
                text = text.replace(self.image_token, "<placeholder>" * thw.prod().long(), 1)
                image_idx += 1
            text = text.replace("<placeholder>", self.image_token)
            assert len(grid_sizes) == image_idx, "Number of images does not match the number of image tokens in the text."

        text_inputs = self.tokenizer(text, **kwargs)
        return text_inputs

    def process_text(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], List[Dict]],
        image_inputs: Dict[str, torch.Tensor] = {},
        return_labels: bool = False,
        **kwargs,
    ):
        kwargs.pop("padding", None)
        kwargs.pop("padding_side", None)

        grid_sizes = []
        for grid_size, merge_size in zip(image_inputs.get("grid_sizes", []), image_inputs.get("merge_sizes", [])):
            if not torch.all(grid_size[1:] % merge_size == 0):
                warnings.warn(f"Grid size {grid_size} is not divisible by merge size. Some undesired errors may occur.")
            if grid_size[0] == 1:
                grid_sizes.append(grid_size[1:] / merge_size)
            elif grid_size[0] > 1:
                grid_sizes.extend([grid_size[1:] / merge_size] * grid_size[0])

        if return_labels:
            return self._process_text_with_label(text, grid_sizes, **kwargs)
        return self._process_text_without_label(text, grid_sizes, **kwargs)

    def process_images(
        self,
        images: ImageInput = None,
        merge_size: Optional[int] = 1,
        **kwargs,
    ):
        if images is None:
            return {}
        image_inputs = self.image_processor(images=images, merge_size=merge_size, **kwargs)
        return image_inputs

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], List[Dict]] = None,
        images: ImageInput = None,
        merge_size: Optional[int] = 1,
        return_labels: bool = False,
        **kwargs: Unpack[Videollama3ProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **grid_sizes** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            Videollama3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        output_kwargs["text_kwargs"].pop("padding", None)
        output_kwargs["text_kwargs"].pop("padding_side", None)

        image_inputs = self.process_images(images, merge_size, **output_kwargs["images_kwargs"])
        text_inputs = self.process_text(text, image_inputs, return_labels, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def _load_multimodal_data(self, conversation: List[Dict[str, Any]]):
        multimodal_info = defaultdict(list)
        new_conversation = []
        for message in conversation:
            new_message = {"role": message["role"]}
            if not isinstance(message["content"], (list, tuple)):
                new_message["content"] = message["content"]
                new_conversation.append(new_message)
                continue

            new_contents = []
            for content in message["content"]:
                if not isinstance(content, dict):
                    new_contents.append(content)
                    continue
                assert "type" in content, "Content must have 'type' field."
                if content["type"] in ["image", "video"] and content["type"] in content and isinstance(content[content["type"]], dict):
                    # TODO: support other types which are not compatible with json
                    load_args = content[content["type"]]
                    data_id = json.dumps({k: v for k, v in load_args.items() if not k in ["start_time", "end_time"]})
                    new_content = copy.deepcopy(content)
                    multimodal_info[data_id].append(new_content)
                    new_contents.append(new_content)
                else:
                    new_contents.append(content)

            new_message["content"] = new_contents
            new_conversation.append(new_message)

        for data_id, contents in multimodal_info.items():
            data_type = contents[0]["type"]
            if data_type == "image":
                image = self.load_images(contents[0][data_type]["image_path"])[0]
                for content in contents:
                    content["image"] = image.copy()

            elif data_type == "video":
                # TODO: start_time is None?
                start_times = [content["video"].get("start_time", 0.) for content in contents]
                end_times = [content["video"].get("end_time", float("inf")) for content in contents]

                load_args = contents[0][data_type]
                start_time, end_time = min(start_times), max(end_times)
                if start_time > 0:
                    load_args["start_time"] = start_time
                if end_time < float("inf"):
                    load_args["end_time"] = end_time
                images, timestamps = self.load_video(**load_args)

                for content, start_time, end_time in zip(contents, start_times, end_times):
                    cur_images, cur_timestamps = [], []
                    for image, timestamp in zip(images, timestamps):
                        if start_time <= timestamp <= end_time:
                            cur_images.append(image.copy())
                            cur_timestamps.append(timestamp)

                    content[data_type] = cur_images
                    content["num_frames"] = len(cur_images)
                    content["timestamps"] = cur_timestamps

        return new_conversation

    def _gather_multimodal_data(self, conversation: List[Dict[str, Any]]):
        images = []
        for message in conversation:
            if not isinstance(message["content"], (list, tuple)):
                continue
            for content in message["content"]:
                if not isinstance(content, dict):
                    continue
                if content["type"] == "video":
                    video = content["video"]
                    assert is_valid_video(video), f"Invalid video data: {video}."
                    images.append(video)
                if content["type"] == "image":
                    image = content["image"]
                    assert is_valid_image(image), f"Invalid image data: {image}."
                    images.append(image)
        images = images if len(images) > 0 else None
        return images

    def apply_chat_template(
        self,
        conversation: List[Dict[str, Any]],
        chat_template: Optional[str] = None,
        tokenize: bool = False,
        add_system_prompt: bool = False,
        add_generation_prompt: bool = False,
        add_think_prompt: bool = False,
        return_dict: bool = False,
        **kwargs,
    ) -> str:
        """
        Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
        conversations to turn them into a single tokenizable string.
        Args:
            conversation (`List[Dict, str, str]`):
                The conversation to format.
            chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the tokenizer's
                chat template is used.
            tokenize (`bool`, *optional*, defaults to `False`):
                Whether to tokenize the output or not.
            add_system_prompt (`bool`, *optional*, defaults to `False`):
                Whether to add the system prompt to the output or not.
            add_generation_prompt (`bool`, *optional*, defaults to `False`):
                Whether to add the generation prompt to the output or not.
            image_token (`Optional[str]`, *optional*, defaults to `<image>`):
                The token to use for indicating images in the conversation.
            **kwargs:
                Additional keyword arguments
        """

        if chat_template is None:
            if self.chat_template is not None:
                chat_template = self.chat_template
            else:
                raise ValueError(
                    "No chat template is set for this processor. Please either set the `chat_template` attribute, "
                    "or provide a chat template as an argument. See "
                    "https://huggingface.co/docs/transformers/main/en/chat_templating for more information."
                )

        images = None
        if return_dict:
            conversation = self._load_multimodal_data(conversation)
            images = self._gather_multimodal_data(conversation)

        prompt = self.tokenizer.apply_chat_template(
            conversation,
            chat_template=chat_template,
            tokenize=tokenize,
            add_system_prompt=add_system_prompt,
            add_generation_prompt=add_generation_prompt,
            add_think_prompt=add_think_prompt,
            image_token=self.image_token,
            **kwargs
        )

        out = {"text": prompt, "images": images}
        if return_dict:
            return out
        return out["text"]

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

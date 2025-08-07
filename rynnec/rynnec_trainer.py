# Adopted from: https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py
import os
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
    TRAINER_STATE_NAME,
)
from transformers.utils import (
    is_sagemaker_mp_enabled,
    logging,
)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_encoder', 'vision_resampler', 'text_hidden_fcs', 'region_encoder', 'grounding_encoder']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            if 'lm_head' in name:
                continue
            lora_module_names.add(name)
    
    return list(lora_module_names)

def safe_save_model_for_hf_trainer(trainer: Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "is_alignment", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        # if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if torch.distributed.get_rank() == 0:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    # last_mm = mm_megabatches[-1]
    # last_lang = lang_megabatches[-1]
    # additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # if len(additional_batch) > 0:
    #     megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class RynnECTrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        print('_get_train_sampler')
        print('world size: ', self.args.world_size * self.args.gradient_accumulation_steps)
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        print('group_by_modality_length...')
        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def update_history_loss_dict(self,outputs):
        if not hasattr(self,'history_loss_dict'):
            self.history_loss_dict = {}
        for name, value in outputs.items():
            if 'loss' in name and name != 'loss':
                if name not in self.history_loss_dict:
                    self.history_loss_dict[name] = value.item()
                else:
                    if value != 0:
                        self.history_loss_dict[name] = value.item()
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            if isinstance(outputs, dict) and 'mask_bce_loss' in outputs:
                loss_dict = {}
                for name,value in outputs.items():
                    if 'loss' in name and name != 'loss':
                        loss_value = value.item()
                        if loss_value == 0 and hasattr(self,'history_loss_dict'):
                            loss_value = self.history_loss_dict[name]
                        loss_dict[name] = loss_value
                self.update_history_loss_dict(outputs)
                self.log(loss_dict)

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss


    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            optimized_parameters = [(n, p) for n, p in opt_model.named_parameters() if p.requires_grad]
            optimizer_grouped_parameters = []

            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            if self.args.llm_lr is not None:
                lm_parameters = [
                    name for name, _ in optimized_parameters if "vision_encoder" not in name and "mm_projector" not in name and "region_encoder" not in name and "grounding_encoder" not in name
                ]
                decay_lm_parameters = [name for name in lm_parameters if name in decay_parameters]
                nodecay_lm_parameters = [name for name in lm_parameters if name not in decay_parameters]
                optimizer_grouped_parameters.extend([
                    {
                        "params": [p for n, p in optimized_parameters if n in decay_lm_parameters],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.llm_lr,
                    },
                    {
                        "params": [p for n, p in optimized_parameters if n in nodecay_lm_parameters],
                        "weight_decay": 0.0,
                        "lr": self.args.llm_lr,
                    }
                ])

            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in optimized_parameters if "mm_projector" in name]
                decay_projector_parameters = [name for name in projector_parameters if name in decay_parameters]
                nodecay_projector_parameters = [name for name in projector_parameters if name not in decay_parameters]
                optimizer_grouped_parameters.extend([
                    {
                        "params": [p for n, p in optimized_parameters if n in decay_projector_parameters], 
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [p for n, p in optimized_parameters if n in nodecay_projector_parameters],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    }
                ])

            if self.args.vision_encoder_lr is not None:
                vision_encoder_parameters = [name for name, _ in optimized_parameters if "vision_encoder" in name]
                decay_vision_encoder_parameters = [name for name in vision_encoder_parameters if name in decay_parameters]
                nodecay_vision_encoder_parameters = [name for name in vision_encoder_parameters if name not in decay_parameters]
                optimizer_grouped_parameters.extend([
                    {
                        "params": [p for n, p in optimized_parameters if n in decay_vision_encoder_parameters], 
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_encoder_lr,
                    },
                    {
                        "params": [p for n, p in optimized_parameters if n in nodecay_vision_encoder_parameters],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_encoder_lr,
                    }
                ])

            if self.args.region_encoder_lr is not None:
                projector_parameters = [name for name, _ in optimized_parameters if "region_encoder" in name]
                decay_projector_parameters = [name for name in projector_parameters if name in decay_parameters]
                nodecay_projector_parameters = [name for name in projector_parameters if name not in decay_parameters]
                optimizer_grouped_parameters.extend([
                    {
                        "params": [p for n, p in optimized_parameters if n in decay_projector_parameters], 
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.region_encoder_lr,
                    },
                    {
                        "params": [p for n, p in optimized_parameters if n in nodecay_projector_parameters],
                        "weight_decay": 0.0,
                        "lr": self.args.region_encoder_lr,
                    }
                ])
            if self.args.sam_decoder_lr is not None:
                projector_parameters = [name for name, _ in optimized_parameters if "grounding_encoder" in name and "image_encoder" not in name]
                decay_projector_parameters = [name for name in projector_parameters if name in decay_parameters]
                nodecay_projector_parameters = [name for name in projector_parameters if name not in decay_parameters]
                optimizer_grouped_parameters.extend([
                    {
                        "params": [p for n, p in optimized_parameters if n in decay_projector_parameters], 
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.sam_decoder_lr,
                    },
                    {
                        "params": [p for n, p in optimized_parameters if n in nodecay_projector_parameters],
                        "weight_decay": 0.0,
                        "lr": self.args.sam_decoder_lr,
                    }
                ])

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'is_alignment', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
            self.args.distributed_state.wait_for_everyone()
        else:
            # NOTE: Supporting save complete lora checkpoint during training.
            if self.args.lora_enable:
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)

                state_dict = get_peft_state_maybe_zero_3(self.model.named_parameters(), self.args.lora_bias)
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters())
           
                # add for qwen2
                if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'lm_head'):
                    lm_head_weight = self.model.base_model.lm_head.weight.cpu() 
                    non_lora_state_dict['base_model.lm_head.weight'] = lm_head_weight
                    print("add base_model.lm_head.weight")
                else:
                    print("The model does not have 'base_model.lm_head.weight' attribute.")


                if self.args.local_rank == 0 or self.args.local_rank == -1:
                    # save for acquring `config.json`
                    self.model.config.save_pretrained(output_dir)
                    # save for acquring `adapter_config.json`, `adapter_model.bin`
                    # self.model.save_pretrained(output_dir, state_dict=state_dict)
                    torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))

                # save for acquring lora adapter parameters & trainer states: `adapter_config.json`, `adapter_model.safetensors`
                super(RynnECTrainer, self)._save_checkpoint(model, trial, metrics)
            else:
                super(RynnECTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'is_alignment', False):
            pass
        else:
            super(RynnECTrainer, self)._save(output_dir, state_dict)

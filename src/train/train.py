import torch
import logging
import torch
import numpy as np
import random

from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
    TrainingArguments,
    ProgressCallback,
    Trainer,
    set_seed,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def set_seed_all(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

    # If you're using CUDA, ensure CUDA determinism
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_all_linear_layers(model):
    """
    Get module that are linear to feed into target module for
    Lora config
    """

    linear_classes = (torch.nn.Linear, torch.nn.Embedding)

    linear_module_names = set()
    for name, module in model.named_modules():
        # match with all linear classes.
        if isinstance(module, linear_classes):
            names = name.rsplit(".", 1)[-1]  # get the base name
            linear_module_names.add(names)

    return list(linear_module_names)


def train(config):
    (
        lora_args,
        tokenizer_args,
        model_args,
        training_args,
        data_args,
        callback_args,
        wandb_args,
        huggingface_args
    ) = (
        config.lora_args,
        config.tokenizer_args,
        config.model_args,
        config.training_args,
        config.data_args,
        config.callback_args,
        config.wandb_args,
        config.huggingface_args,
    )

    compute_dtype = (
        torch.float16
        if getattr(training_args, "fp16", None)
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
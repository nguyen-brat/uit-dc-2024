import torch
import torch
import numpy as np
import random
import re

from torch.utils.data import DataLoader
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLProcessor, AutoProcessor

from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
    TrainingArguments,
    ProgressCallback,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from ..dataloader import MSDDataloader, MSDDataCollator
from ..model import MSD, MSDConfig, apply_liger_kernel_to_msd


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
            linear_module_names.add(name)

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

    if getattr(training_args, "seed", None):
        set_seed_all(training_args.seed)

    ################### Load data
    min_pixels=data_args.pop("min_pixels", None)
    max_pixels=data_args.pop("max_pixels", None)
    if min_pixels and max_pixels:
        processor = AutoProcessor.from_pretrained(
            model_args.base_model,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    else:
        processor = AutoProcessor.from_pretrained(model_args.base_model)
    train_data_path = data_args.pop("train_data", None)
    if getattr(data_args, "val_data", None):
        val_data_path = data_args.pop("val_data", None)
        val_dataloader = MSDDataloader(
            **val_data_path,
            **data_args,
        )

    train_dataloader = MSDDataloader(
        **train_data_path,
        **data_args,
    )

    collator = MSDDataCollator(processor, padding=True)

    ################### Load model
    if model_args.base_model:
        quantization_config=None
        if training_args.quantization == 4:
            quantization_4bit_config = {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "fp4",
                "bnb_4bit_compute_dtype": compute_dtype,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_storage": compute_dtype,
            }
            quantization_config = BitsAndBytesConfig(**quantization_4bit_config)
        elif training_args.quantization == 8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            if training_args.quantization:
                raise ValueError(f"Not support lora quantization in {training_args.quantization} bit")

        if training_args.quantization:
            config = MSDConfig(
                quantization_config = quantization_config,
                torch_dtype=compute_dtype,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                **model_args
            )
        else:
            config = MSDConfig(
                torch_dtype=compute_dtype,
                **model_args
            )
        model = MSD(config)
        model.freeze_vision()
        if training_args.pop("freeze_base", None):
            model.freeze_base()
        ################ *****************************************************
        if training_args.pop("use_liger_kernel", None):
            apply_liger_kernel_to_msd(fused_linear_cross_entropy=False, model=model)
        if training_args.quantization:
            model = prepare_model_for_kbit_training(model)
        training_args.pop("quantization", None)
    else:
        raise ValueError("You must specify embedder_based and thinker_based")
    

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(training_args.gradient_checkpointing_kwargs)
        model.config.use_cache = False
        model.enable_input_require_grads()

    if training_args.pop("use_lora", None):
        target_modules = lora_args.pop("target_modules")
        if target_modules == "all-linear":
            target_modules = get_all_linear_layers(model)
        modules_to_save = [string for string in target_modules if any(re.match(pattern, string) for pattern in lora_args.pop("modules_to_save", []))]
        
        lora_config = LoraConfig(
            target_modules=[module for module in target_modules if module not in modules_to_save],
            modules_to_save=modules_to_save,
            **lora_args
        )
        # print(target_modules)
        if lora_args.modules_to_save == []:
            lora_args.pop("modules_to_save", None)
        model = get_peft_model(model, lora_config)

    # ############################################################ TRAINER
    train_args = TrainingArguments(
        # other args and kwargs here
        **training_args,
        **wandb_args,
        **huggingface_args,
    )

    # Initialize our Trainer

    # Filter out the ProgressCallback

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=collator,
        train_dataset=train_dataloader,
        eval_dataset=val_dataloader if training_args.do_eval else None,
    )
    if training_args.do_eval:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=callback_args.patient))

    # Training
    checkpoint = False
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    if trainer.is_local_process_zero:
        trainer.save_state()
        trainer.save_model()
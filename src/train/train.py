import torch
import torch
import numpy as np
import random
import re
import os
import json
from tqdm import tqdm

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

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
import evaluate
from ..dataloader import MSDDataloader, MSDDataCollator
from ..model import MSD, MSDConfig, apply_liger_kernel_to_msd
from transformers import Trainer


def inference(
        model,
        phase = "dev",
        annotation_path = "data/warn_up/ocr_llm.json",
        image_path = "data/warn_up/warmup-images",
        output_dir = "submit/dump_results.json",
):
    result = {
        "results": {},
        "phase": phase,
    }
    with open(annotation_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    for id, value in tqdm(data.items()):
        caterory = model.predict(value, image_path)
        result["results"][id] = caterory

        with open(output_dir, "w", encoding="utf") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)


def calculate_f1_scores(y_true, y_pred):
    label_names = {
        0: 'multi-sarcasm',
        1: 'not-sarcasm',
        2: 'image-sarcasm',
        3: 'text-sarcasm'
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    f1_scores = {}
    for idx in range(4):
        tp = cm[idx, idx]
        fp = np.sum(cm[:, idx]) - tp
        fn = np.sum(cm[idx, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores[label_names[idx]] = f1
    
    return f1_scores


metric = evaluate.load("f1")
def compute_metrics(eval_pred):
    output, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(output, axis=-1)
    confusion_output = calculate_f1_scores(labels, predictions)
    return dict(
        multi_sarcasm_f1=confusion_output["multi-sarcasm"],
        not_sarcasm_f1=confusion_output["not-sarcasm"],
        text_sarcasm_f1=confusion_output["text-sarcasm"],
        image_sarcasm_f1=confusion_output["image-sarcasm"],
        f1=f1_score(labels, predictions, average='macro')
    )
    #return metric.compute(predictions=predictions, references=labels, average="macro")

def set_seed_all(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

    # If you're using CUDA, ensure CUDA determinism
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_all_linear_modules(model):
    """
    Get module that are linear to feed into target module for
    Lora config
    """

    linear_classes = torch.nn.Linear

    linear_module_names = set()
    for name, module in model.named_modules():
        # match with all linear classes.
        if isinstance(module, linear_classes):
            # names = name.rsplit(".", 1)[-1]  # get the base name
            linear_module_names.add(name)

    return list(linear_module_names)

def get_all_linear_layers(model, freeze_vision_tower: bool):
    r"""
    Finds all available modules to apply lora or galore.
    """
    model_type = getattr(model.config, "model_type", None)
    forbidden_modules = {"lm_head"}
    if model_type == "chatglm":
        forbidden_modules.add("output_layer")
    elif model_type == "internlm2":
        forbidden_modules.add("output")
    elif model_type in ["llava", "llava_next", "llava_next_video", "paligemma", "video_llava"]:
        forbidden_modules.add("multi_modal_projector")
    elif (model_type == "qwen2_vl") and (model_type == "msd" ):
        forbidden_modules.add("merger")
        forbidden_modules.add("encoder_layers")
        forbidden_modules.add("classification_layer")

    if freeze_vision_tower:
        if (model_type == "qwen2_vl") and (model_type == "msd" ):
            forbidden_modules.add("visual")
        else:
            forbidden_modules.add("vision_tower")

    module_names = set()
    for name, module in model.named_modules():
        if any(forbidden_module in name for forbidden_module in forbidden_modules):
            continue

        if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__:
            module_names.add(name.split(".")[-1])

    return list(module_names)



def train(config):
    (
        lora_args,
        model_args,
        training_args,
        data_args,
        callback_args,
        wandb_args,
        huggingface_args
    ) = (
        config.lora_args,
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

    collator = MSDDataCollator(processor, padding=True, max_length=data_args.pop("max_length", None))
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
    
    # print(model)
    # if train gradient checkpoint must enable input require_grad
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(training_args.gradient_checkpointing_kwargs)
        model.config.use_cache = False
        model.enable_input_require_grads()

    use_lora = training_args.pop("use_lora", None)
    if use_lora:
        target_modules = lora_args.pop("target_modules", [])
        modules_to_save = lora_args.pop("modules_to_save", [])
        if target_modules == "all-linear":
            target_modules = get_all_linear_layers(model, freeze_vision_tower=True)
        else:
            all_linear_module_names = get_all_linear_modules(model)
            list_modules_to_save = [
                module for module in all_linear_module_names
                if any(re.match(pattern, module) for pattern in modules_to_save)
            ]
            list_target_modules = [
                module for module in all_linear_module_names if
                any(re.match(pattern, module) for pattern in target_modules)
                and module not in modules_to_save
            ]

        lora_config = LoraConfig(
            target_modules=list_target_modules,
            modules_to_save=list_modules_to_save,
            **lora_args
        )
        model = get_peft_model(model, lora_config)

    # ############################################################ TRAINER
    train_args = TrainingArguments(
        # other args and kwargs here
        **training_args,
        **wandb_args,
        **huggingface_args,
    )


    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=collator,
        train_dataset=train_dataloader,
        eval_dataset=val_dataloader if training_args.do_eval else None,
        compute_metrics=compute_metrics # if training_args.do_eval else None,
    )
    if training_args.do_eval and callback_args.get("patient", None):
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=callback_args.patient))

    # Training
    checkpoint = False
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    if trainer.is_local_process_zero() and use_lora:
        model = trainer.model.merge_and_unload()
        model.save_pretrained(
            f"{training_args.output_dir}/merged_model",
            safe_serialization=True,
            max_shard_size="4GB"  # Adjust this value based on your needs
        )
        processor.save_pretrained(f"{training_args.output_dir}/merged_model")
        chat_template = processor.tokenizer.chat_template
        with open(os.path.join(f"{training_args.output_dir}/merged_model", 'chat_template.json'), 'w') as f:
            json.dump({'chat_template': chat_template}, f, indent=2)
        
        
        
        # val_dataset = DataLoader(val_dataloader, collate_fn=collator, batch_size=2)
        # output = trainer.evaluation_loop(val_dataset, description="evaluation testing when train end", metric_key_prefix="f1")
        # print("=========================================")
        # print(output)
        # print("*********************")
        # print(output.metrics)
        # pass
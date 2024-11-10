import json
import re
import torch
import os
import argparse
import itertools
from functools import partial
from os.path import join as osp
from tqdm import tqdm
import torch.multiprocessing as mp
from transformers import (
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    LlavaForConditionalGeneration,
    AutoProcessor,
    AutoModel,
    AutoTokenizer,
)
from prompt import (
    create_reas_prompt_pixtral,
    create_reas_prompt_qwen2_vl,
    create_vi_intern_prompt,
    create_vi_intern_prompt_image
)
import traceback
from datetime import datetime
import gc

def split_dict(data, chunks):
    """Split dictionary into n chunks, handling uneven sizes
    
    Args:
        data (dict): Dictionary to split
        chunks (int): Number of chunks to split into
        
    Returns:
        list: List of dictionaries, where the last chunk may have more items if uneven
    """
    items = list(data.items())
    base_chunk_size = len(items) // chunks
    remainder = len(items) % chunks
    
    result = []
    start = 0
    
    for i in range(chunks):
        # Add one extra item to some chunks if there's a remainder
        current_chunk_size = base_chunk_size + (1 if i < remainder else 0)
        end = start + current_chunk_size
        
        chunk_dict = {k: v for k, v in items[start:end]}
        result.append(chunk_dict)
        start = end
    
    return result

def get_n_items_at_a_time(dictionary, n):
    dict_items = iter(dictionary.items())
    while True:
        items = dict(itertools.islice(dict_items, n))
        if not items:
            break
        yield items

def group_subitems_by_key(batch):
    images = []
    captions = []
    labels = []
    ocrs = []
    keys = []

    for key, value in batch.items():
        images.append(value['image'])
        captions.append(value['caption'])
        labels.append(value['label'])
        ocrs.append(value.get('ocr', None))
        keys.append(key)

    return {
        'keys': keys,
        'images': images,
        'captions': captions,
        'labels': labels,
        'ocrs': ocrs
    }

def clean_hashtag(caption):
    return re.sub(r'\n?#\S+', '', caption).strip()

class ReasoningPrompt:
    def __init__(
            self, 
            model="pixtral",
            generation_config=None,
            gpu_id=0
    ):
        self.loaded_model = model
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)
        if model == "pixtral":
            self.load_pixtral()
        elif model == "qwen2":
            self.load_qwen2()
        elif model == "vi_intern":
            self.load_vi_intern()
        elif model == "vi_intern_image":
            self.load_vi_intern_image()
        else:
            raise KeyError(f"There currently not support {model} only support qwen2, pixtral, vi_intern, vi_intern_image")
        self.generation_config = dict(
            max_new_tokens=512, do_sample=False, num_beams=3, repetition_penalty=2.0
        ) if generation_config is None else generation_config

    @torch.no_grad()
    def generation(self, image_path, caption, label, ocr)->str:
        inputs = self.prompt_creator(image_paths=image_path, captions=caption, labels=label, ocrs=ocr)
        if self.loaded_model not in ["vi_intern", "vi_intern_image"]:
            inputs = inputs.to(self.model.device)
            generated_ids = self.model.generate(**inputs, **self.generation_config)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            del inputs
            del generated_ids
            torch.cuda.empty_cache()
            gc.collect()
            return output_text
        else:
            response, _ = self.model.chat(**inputs, generation_config=self.generation_config, history=None, return_history=True)
            return [response]

    def load_pixtral(self):
        self.model = LlavaForConditionalGeneration.from_pretrained(
            "mistral-community/pixtral-12b",
            device_map=f"cuda:{self.gpu_id}",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        ).eval()
        self.processor = AutoProcessor.from_pretrained("mistral-community/pixtral-12b")
        self.processor.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        self.prompt_creator = partial(create_reas_prompt_pixtral, self.processor)
        print(f"Loading pixtral model on GPU {self.gpu_id}")

    def load_qwen2(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype="auto",
            device_map=f"cuda:{self.gpu_id}"
        ).eval()
        self.processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        self.prompt_creator = partial(create_reas_prompt_qwen2_vl, self.processor)
        print(f"Loading qwen2-vl model on GPU {self.gpu_id}")
    
    def load_vi_intern(self):
        self.model = AutoModel.from_pretrained(
            "5CD-AI/Vintern-3B-beta",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().to(f"cuda:{self.gpu_id}")
        tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vintern-3B-beta", trust_remote_code=True, use_fast=False)
        self.prompt_creator = partial(create_vi_intern_prompt, tokenizer)
        print(f"Loading Vintern-3B-beta on GPU {self.gpu_id}")

    def load_vi_intern_image(self):
        self.model = AutoModel.from_pretrained(
            "5CD-AI/Vintern-3B-beta",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().to(f"cuda:{self.gpu_id}")
        tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vintern-3B-beta", trust_remote_code=True, use_fast=False)
        self.prompt_creator = partial(create_vi_intern_prompt_image, tokenizer)
        print(f"Loading Vintern-3B-beta on GPU {self.gpu_id}")

def process_chunk(gpu_id, data_chunk, model_name, image_path, output_path, batch_size):
    output_file_name = output_path.split("/")[-1].split(".")[0]
    result = {key: {} for key in data_chunk.keys()}
    error_keys = []
    
    prompt_creator = ReasoningPrompt(model_name, gpu_id=gpu_id)
    
    total_batches = (len(data_chunk) + batch_size - 1) // batch_size
    for batch in tqdm(get_n_items_at_a_time(data_chunk, batch_size), 
                     total=total_batches, 
                     desc=f'GPU {gpu_id}'):
        group_item = group_subitems_by_key(batch)
        try:
            reasonings = prompt_creator.generation(
                image_path=[osp(image_path, image) for image in group_item["images"]],
                caption=[clean_hashtag(caption) for caption in group_item["captions"]],
                label=group_item["labels"],
                ocr=group_item["ocrs"],
            )

            for key, reasoning in zip(group_item["keys"], reasonings):
                if model_name == "vi_intern_image":
                    result[key]["image_reasoning"] = reasoning
                else:
                    result[key]["reasoning"] = reasoning
                result[key].update(batch[key])

            # Save intermediate results
            intermediate_path = f"{output_path}_gpu{gpu_id}_intermediate.json"
            with open(intermediate_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)

        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            error_message = f"{timestamp} - GPU {gpu_id} - Error: {str(e)}\n"
            error_message += traceback.format_exc()
            with open(f"log/error_log_{output_file_name}_gpu{gpu_id}.txt", "a") as error_file:
                error_file.write(error_message + "\n")
            
            error_keys.extend(group_item["keys"])
            with open(f"log/key_error_{output_file_name}_gpu{gpu_id}.txt", "a") as key_error_file:
                for key in error_keys:
                    key_error_file.write(f"{key}\n")

    # Save final results for this GPU
    final_path = f"{output_path}_gpu{gpu_id}_final.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vi_intern")
    parser.add_argument("--image_path", type=str, default="data/public_train/train-images")
    parser.add_argument("--input_path", type=str, default="data/public_train/vimmsd-train_01.json")
    parser.add_argument("--output_path", type=str, default="data/public_train/reasoning_vi_intern_image_reasoning_01.json")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    # Load and split data
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # continue generating
    if os.path.isfile(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        for key in result.keys():
            if (result[key] != {}) and (key in data.keys()):
                del data[key]
    else:
        result = {key:{} for key in data.keys()}

    data_chunks = split_dict(data, 2)
    
    # Create processes for both GPUs
    processes = []
    for gpu_id in range(2):
        p = mp.Process(
            target=process_chunk,
            args=(
                gpu_id,
                data_chunks[gpu_id],
                args.model,
                args.image_path,
                args.output_path,
                args.batch_size
            )
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Merge results from both GPUs
    final_results = {}
    for gpu_id in range(2):
        with open(f"{args.output_path}_gpu{gpu_id}_final.json", "r", encoding="utf-8") as f:
            gpu_results = json.load(f)
            final_results.update(gpu_results)

    final_results = {**result, **final_results}
    # Save final merged results
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    
    # Cleanup intermediate files
    for gpu_id in range(2):
        for file_type in ['intermediate', 'final']:
            file_path = f"{args.output_path}_gpu{gpu_id}_{file_type}.json"
            if os.path.exists(file_path):
                os.remove(file_path)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
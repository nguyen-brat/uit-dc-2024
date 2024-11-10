# import numpy as np
# import torch
# import torchvision.transforms as T
# # from decord import VideoReader, cpu
# from PIL import Image
# from torchvision.transforms.functional import InterpolationMode
# from transformers import AutoModel, AutoTokenizer

# from utils import load_image

# model_name = "5CD-AI/Vintern-3B-beta" # "5CD-AI/Vintern-4B-v1"
# model = AutoModel.from_pretrained(
#     model_name,
#     attn_implementation="flash_attention_2",
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
# ).eval().cuda()
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
# generation_config = dict(max_new_tokens= 512, do_sample=False, num_beams = 3, repetition_penalty=3.5)
# # question = "<image>\nTrích xuất văn bản có trong ảnh."
# question = '<image>\nMô tả hình ảnh một cách chi tiết.'

# ################################

# import json
# from tqdm import tqdm
# from os.path import join as osp
# import os
# import sys
# import argparse

# script_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
# grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
# sys.path.extend([parent_dir, grand_dir])

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--annotation_path", type=str, default="data/public_train/vimmsd-train_02.json")
#     parser.add_argument("--image_folder_path", type=str, default="data/public_train/train-images")
#     parser.add_argument("--output_path", type=str, default="data/public_train/ocr_llm_02.json")
#     args = parser.parse_args()

#     annotation_path = args.annotation_path
#     image_folder_path = args.image_folder_path
#     output_path = args.output_path

#     with open(annotation_path, "r", encoding='utf-8') as f:
#         data = json.load(f)

#     for key, value in tqdm(data.items()):
#         image_path = osp(image_folder_path, value["image"])
#         pixel_values = load_image(image_path, max_num=6).to(torch.bfloat16).to(model.device)
#         response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
#         data[key]["ocr"] = response
#         with open(output_path, "w", encoding='utf-8') as f:
#             json.dump(data, f, indent=4, ensure_ascii=False)

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from utils import load_image
import json
from tqdm import tqdm
from os.path import join as osp
import os
import sys
import argparse
import torch.multiprocessing as mp
from itertools import islice

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

def process_chunk(rank, data_chunk, model_name, image_folder_path, output_path, question):
    """Process a chunk of data on specified GPU"""
    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    print(f"Process {rank} using GPU {device}")
    
    # Load model on specific GPU
    model = AutoModel.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval().to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=512, do_sample=False, num_beams=3, repetition_penalty=3.5)
    
    # Process images
    results = {}
    for key, value in tqdm(data_chunk.items(), desc=f'GPU {rank}'):
        image_path = osp(image_folder_path, value["image"])
        pixel_values = load_image(image_path, max_num=6).to(torch.bfloat16).to(device)
        response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
        results[key] = {"image": value["image"], "ocr": response}
        
        # Save intermediate results periodically
        if len(results) % 10 == 0:  # Save every 10 processed items
            intermediate_path = f"{output_path}_gpu{rank}_intermediate.json"
            with open(intermediate_path, "w", encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
    
    # Save final results for this GPU
    final_path = f"{output_path}_gpu{rank}_final.json"
    with open(final_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

def run_parallel_processing(gpu_id, data_chunk, model_name, image_folder_path, output_path, question):
    """Wrapper function for parallel processing"""
    process_chunk(gpu_id, data_chunk, model_name, image_folder_path, output_path, question)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", type=str, default="data/warn_up/ocr.json")
    parser.add_argument("--image_folder_path", type=str, default="data/warn_up/warmup-images")
    parser.add_argument("--output_path", type=str, default="data/warn_up/dump_output.json")
    args = parser.parse_args()
    
    # Load data
    with open(args.annotation_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    # Split data into two chunks
    data_chunks = split_dict(data, 2)
    
    model_name = "5CD-AI/Vintern-3B-beta"
    question = '<image>\nMô tả hình ảnh một cách chi tiết.'
    
    # Create processes for both GPUs
    processes = []
    for gpu_id in range(2):
        p = mp.Process(
            target=run_parallel_processing,
            args=(gpu_id, data_chunks[gpu_id], model_name, args.image_folder_path, args.output_path, question)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Merge results from both GPUs
    results = {}
    for gpu_id in range(2):
        with open(f"{args.output_path}_gpu{gpu_id}_final.json", "r", encoding='utf-8') as f:
            gpu_results = json.load(f)
            results.update(gpu_results)
    
    # Save final merged results
    with open(args.output_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    # Cleanup intermediate files
    for gpu_id in range(2):
        for file_type in ['intermediate', 'final']:
            file_path = f"{args.output_path}_gpu{gpu_id}_{file_type}.json"
            if os.path.exists(file_path):
                os.remove(file_path)

    print("Processing completed and results merged successfully!")

if __name__ == "__main__":
    # Set multiprocessing method
    mp.set_start_method('spawn', force=True)
    main()
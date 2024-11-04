import numpy as np
import torch
import torchvision.transforms as T
# from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from utils import load_image

model_name = "5CD-AI/Vintern-3B-beta" # "5CD-AI/Vintern-4B-v1"
model = AutoModel.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
generation_config = dict(max_new_tokens= 512, do_sample=False, num_beams = 3, repetition_penalty=3.5)
# question = "<image>\nTrích xuất văn bản có trong ảnh."
question = '<image>\nMô tả hình ảnh một cách chi tiết.'

################################

import json
from tqdm import tqdm
from os.path import join as osp
import os
import sys
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
sys.path.extend([parent_dir, grand_dir])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", type=str, default="data/public_train/vimmsd-train_02.json")
    parser.add_argument("--image_folder_path", type=str, default="data/public_train/train-images")
    parser.add_argument("--output_path", type=str, default="data/public_train/ocr_llm_02.json")
    args = parser.parse_args()

    annotation_path = args.annotation_path
    image_folder_path = args.image_folder_path
    output_path = args.output_path

    with open(annotation_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    for key, value in tqdm(data.items()):
        image_path = osp(image_folder_path, value["image"])
        pixel_values = load_image(image_path, max_num=6).to(torch.bfloat16).to(model.device)
        response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
        data[key]["ocr"] = response
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
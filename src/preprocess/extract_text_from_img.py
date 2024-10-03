import json
from os.path import join as osp
import os
import sys
from tqdm import tqdm
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
sys.path.extend([parent_dir, grand_dir])

from src.ocr.ocr import ImageOCR

reader = ImageOCR(
    model_path = 'model/ocr/transformerocr.pth',
    model_type = 'model/ocr/vgg-transformer.yml',
    device = 'cuda:0',
    script_path='script/text_detection.sh',
)

with open(r'data/vimmsd-warmup.json', "r", encoding='utf-8') as f:
    data = json.load(f)

for key, value in tqdm(data.items()):
    image_path = osp("data/warmup-images", value["image"])
    image = Image.open(image_path)
    output = reader.predict(image, min_samples=1)
    data[key]["ocr"] = output

with open(r"data/ocr.json", "w", encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
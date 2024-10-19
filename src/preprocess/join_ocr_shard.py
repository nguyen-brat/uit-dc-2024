import json
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
sys.path.extend([parent_dir, grand_dir])

ocr_file_1 = "data/public_test/ocr_llm_01.json"
ocr_file_2 = "data/public_test/ocr_llm_02.json"

with open(ocr_file_1, "r", encoding='utf-8') as f:
    ocr_1 = json.load(f)
with open(ocr_file_2, "r", encoding='utf-8') as f:
    ocr_2 = json.load(f)

merge_ocr = ocr_1 | ocr_2

with open("data/public_test/ocr_llm.json", "w", encoding="utf-8") as f:
    json.dump(merge_ocr, f, ensure_ascii="False", indent=4)
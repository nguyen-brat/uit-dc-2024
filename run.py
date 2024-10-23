import argparse
import json
from tqdm import tqdm

from src.model import MSD

def inference(args, model):
    result = {
        "results": {},
        "phase": args.phase,
    }
    with open(args.annotation_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    for id, value in tqdm(data.items()):
        caterory = model.predict(value, args.image_path)
        result["results"][id] = caterory

    with open(args.output_dir, "w", encoding="utf") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model/trained/qwen2_vl_cls_no_base_reas")
    parser.add_argument("--annotation_path", type=str, default="data/public_test/ocr_llm_fix.json")
    parser.add_argument("--image_path", type=str, default="data/public_test/dev-images")
    parser.add_argument("--output_dir", type=str, default="submit/results.json")
    parser.add_argument("--phase", type=str, default="dev")
    args = parser.parse_args()

    model = MSD.from_pretrained(args.model_path, torch_dtype="auto", device_map="auto").eval()
    inference(args, model)
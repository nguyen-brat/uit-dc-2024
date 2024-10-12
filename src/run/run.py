import argparse
import json
from ..model import MSD, MSDConfig

def inference(args, model):
    result = {
        "results": {},
        "phase": args.phase,
    }
    with open(args.data_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    for id, value in data.items():
        caterory = model.predict(value)
        result["results"][id] = caterory

    with open(args.output_dir, "w", encoding="utf") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--phase", type=str, default="dev")
    args = parser.parse_args()

    model = MSD.from_pretrained(args.model_path, torch_type="bf16", device_map="auto").eval()
    inference(args, model)
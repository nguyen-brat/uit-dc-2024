import argparse
import json
from tqdm import tqdm

from src.model import MSD, MSDConfig
import torch
from peft import PeftModel
from accelerate import dispatch_model, infer_auto_device_map

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
    parser.add_argument("--model_path", type=str, default="model/trained/qwen2_vl_cls_qwen2_reason_base")
    parser.add_argument("--annotation_path", type=str, default="data/public_test/ocr_llm_fix.json")
    parser.add_argument("--image_path", type=str, default="data/public_test/dev-images")
    parser.add_argument("--output_dir", type=str, default="submit/dump_results.json")
    parser.add_argument("--phase", type=str, default="dev")
    args = parser.parse_args()


    # model = MSD.from_pretrained(args.model_path, torch_dtype="auto", device_map="auto").eval()
    # print(f'model dtype is: {model.dtype}')
    # config = MSDConfig(extra_layers=0, torch_dtype=torch.bfloat16, device_map="auto")
    # model = MSD._from_config(config)# .to(torch.bfloat16)
    # model.load_adapter("model/adapter/qwen2_vl_cls_qwen2_reason_base/checkpoint-1011")# .cuda()
    model = MSD.from_pretrained("model/hf/qwen2_vl_cls_qwen2_reason_base/merged_model", torch_dtype="auto", device_map="auto").eval()
    # device_map = infer_auto_device_map(
    #     model,
    #     max_memory={
    #         0: "23GiB",  # Adjust based on your GPU memory
    #         1: "23GB",
    #         "cpu": "32GiB"
    #     }
    # )
    # model = dispatch_model(model, device_map=device_map)
    # peft_model = PeftModel.from_pretrained(model, model_id=args.adapter_path, device_map="auto")

    inference(args, model)
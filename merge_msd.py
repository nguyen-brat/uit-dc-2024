from src.model import MSD, MSDConfig
from peft import get_peft_model, PeftModel
import torch
from transformers import AutoProcessor

def merge(args):

    config = MSDConfig(
        extra_layers = args.extra_layers,
        torch_dtype = torch.bfloat16,
    )
    base_model = MSD(config)
    peft_model = PeftModel.from_pretrained(base_model, model_id="model/adapter/qwen2_vl_cls_qwen2_reason_base")
    peft_model = peft_model.merge_and_unload()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    processor.save_pretrained("model/trained/qwen2_vl_cls_qwen2_reason_base")
    peft_model.save_pretrained("model/trained/qwen2_vl_cls_qwen2_reason_base")

if __name__ == "__main__":
    config = MSDConfig(
        base_model = "Qwen/Qwen2-VL-2B-Instruct",
        extra_layers = 4,
        num_class=3,
        torch_dtype = torch.bfloat16,
    )
    base_model = MSD(config).to(torch.bfloat16)
    peft_model = PeftModel.from_pretrained(base_model, model_id="model/hf/4_extra_layer_2b_ce_weight_loss_draft_text_image_reasoning_sum_mask_remove_text_labels_upsampling_image_batch_1/checkpoint-1320")
    peft_model = peft_model.merge_and_unload()
    peft_model.save_pretrained("model/hf/4_extra_layer_2b_ce_weight_loss_draft_text_image_reasoning_sum_mask_remove_text_labels_upsampling_image_batch_1/merged_model")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    processor.save_pretrained("model/hf/4_extra_layer_2b_ce_weight_loss_draft_text_image_reasoning_sum_mask_remove_text_labels_upsampling_image_batch_1/merged_model")
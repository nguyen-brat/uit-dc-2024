DIR=`pwd`
model_path=${DIR}/LLaMA-Factory/models/qwen2_vl_lora_sft_v1
annotation_path=${DIR}/data/public_test/ocr_llm_fix.json
image_path=${DIR}/data/public_test/dev-images
output_dir=${DIR}/submit/result_cls.json

python ${DIR}/run.py --model_path $model_path \
                     --annotation_path $annotation_path \
                     --image_path $image_path \
                     --output_dir $output_dir
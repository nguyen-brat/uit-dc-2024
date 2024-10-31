DIR=`pwd`
model_path=${DIR}/model/hf/qwen2_vl_cls_qwen2_reason_base_focal_loss/merged_model
annotation_path=${DIR}/data/public_test/ocr_llm_fix.json
image_path=${DIR}/data/public_test/dev-images
output_dir=${DIR}/submit/result.json

python ${DIR}/run.py --model_path $model_path \
                     --annotation_path $annotation_path \
                     --image_path $image_path \
                     --output_dir $output_dir \
                     --batch_size 1
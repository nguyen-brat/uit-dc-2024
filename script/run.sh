DIR=`pwd`
model_path=${DIR}/model/hf/6_extrac_layer_freeze_base_ce_weight_loss_remove_text_labels_upsampling_image_batch_1/merged_model
annotation_path=${DIR}/data/public_test/ocr_llm_fix.json
image_path=${DIR}/data/public_test/dev-images
output_dir=${DIR}/submit/results.json

python ${DIR}/run.py --model_path $model_path \
                     --annotation_path $annotation_path \
                     --image_path $image_path \
                     --output_dir $output_dir \
                     --batch_size 1
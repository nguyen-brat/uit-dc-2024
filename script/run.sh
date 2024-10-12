DIR=`pwd`
model_path=${DIR}/dump_shit
annotation_path=${DIR}/data/warn_up/ocr_llm.json
image_path=${DIR}/data/warn_up/warmup-images
output_dir=${DIR}/dump_output.json

python ${DIR}/run.py --model_path $model_path \
                     --annotation_path $annotation_path \
                     --image_path $image_path \
                     --output_dir $output_dir
# process data for public train
ANNOTATION_PATH="data/public_train/vimmsd-train.json"
IMAGE_PATH="data/public_train/train-images"
OUTPUT_PATH="data/public_train/image_text_reasoning/image_text_ocr_llm_reasoning_v2.json"
OUTPUT_IMAGE_UPSAMPLING_PATH="data/public_train/image_text_reasoning/train_image_umsaple_x6_image_text_ocr_llm_reasoning_v2.json"
OUTPUT_TEXT_UPSAMPLING_PATH="data/public_train/image_text_reasoning/train_text_umsaple_x10_image_text_ocr_llm_reasoning_v2.json"
OUTPUT_PATH_OCR="data/public_train/image_text_reasoning/ocr_llm.json"
OUTPUT_PATH_REASONING="data/public_train/image_text_reasoning/reasoning.json"
OUTPUT_PATH_TEXT_REASONING="data/public_train/image_text_reasoning/text_reasoning.json"
OUTPUT_PATH_IMAGE_REASONING="data/public_train/image_text_reasoning/image_reasoning.json"
# prepare ocr using vi_intern
echo "prepare ocr using vi_intern"
python src/preprocess/extract_text_llm.py --annotation_path $ANNOTATION_PATH \
                                          --image_folder_path $IMAGE_PATH \
                                          --output_path $OUTPUT_PATH_OCR
# prepare reasoning
echo "prepare reasoning using both caption and image"
python src/agent/reasoning.py --model "vi_intern" \
                              --image_path $IMAGE_PATH \
                              --input_path $ANNOTATION_PATH \
                              --output_path $OUTPUT_PATH_REASONING \
                              --batch_size 1
# prepare text reasoning
echo "prepare text reasoning"
python src/agent/text_reasoning.py --model "Qwen/Qwen2-7B-Instruct" \
                                   --input_path $ANNOTATION_PATH \
                                   --output_path $OUTPUT_PATH_TEXT_REASONING \
                                   --batch 1
# prepare image reasoning
echo "prepare image reasoning"
python src/agent/reasoning.py --model "vi_intern_image" \
                              --image_path $IMAGE_PATH \
                              --input_path $ANNOTATION_PATH \
                              --output_path $OUTPUT_PATH_IMAGE_REASONING \
                              --batch_size 1

# join multiple reasoning file
python src/preprocess/join_multiple_reasoning.py --text_reasoning_path $OUTPUT_PATH_TEXT_REASONING \
                                                 --image_reasoning_path $OUTPUT_PATH_IMAGE_REASONING \
                                                 --reasoning_path $OUTPUT_PATH_REASONING \
                                                 --ocr_path $OUTPUT_PATH_OCR \
                                                 --input_path $ANNOTATION_PATH \
                                                 --output_path $OUTPUT_PATH

# prepare text upsampling data
python src/preprocess/upsampling_llm.py --label_type "text-sarcasm" \
                                        --num_sample_used 61 \
                                        --num_upsample 10 \
                                        --input_path $OUTPUT_PATH \
                                        --output_path $OUTPUT_TEXT_UPSAMPLING_PATH

# prepare image umsampling data
python src/preprocess/upsampling_llm.py --label_type "image-sarcasm" \
                                        --num_sample_used 404 \
                                        --num_upsample 6 \
                                        --input_path $OUTPUT_PATH \
                                        --output_path $OUTPUT_IMAGE_UPSAMPLING_PATH

# prepare share GPU format for pretrained vision-LLM
python src/preprocess/prepare_share_gpt_data.py

####################### process data for privat test
ANNOTATION_PATH="data/private_test/vimmsd-private-test.json"
IMAGE_PATH="data/private_test/test-images"
OUTPUT_PATH="data/private_test/processed_data.json"
OUTPUT_PATH_OCR="data/private_test/ocr_llm.json"
OUTPUT_PATH_REASONING="data/private_test/reasoning.json"
OUTPUT_PATH_TEXT_REASONING="data/private_test/text_reasoning.json"
OUTPUT_PATH_IMAGE_REASONING="data/private_test/image_reasoning.json"
# prepare ocr using vi_intern
python src/preprocess/extract_text_llm.py --annotation_path $ANNOTATION_PATH \
                                          --image_folder_path $IMAGE_PATH \
                                          --output_path $OUTPUT_PATH_OCR
# prepare reasoning
python src/agent/reasoning.py --model "vi_intern" \
                              --image_path $IMAGE_PATH \
                              --input_path $ANNOTATION_PATH \
                              --output_path $OUTPUT_PATH_REASONING \
                              --batch_size 1
# prepare text reasoning
python src/agent/text_reasoning.py --model "Qwen/Qwen2-7B-Instruct" \
                                   --input_path $ANNOTATION_PATH \
                                   --output_path $OUTPUT_PATH_TEXT_REASONING \
                                   --batch 1
# prepare image reasoning
python src/agent/reasoning.py --model "vi_intern_image" \
                              --image_path $IMAGE_PATH \
                              --input_path $ANNOTATION_PATH \
                              --output_path $OUTPUT_PATH_IMAGE_REASONING \
                              --batch_size 1

# join multiple reasoning file
python src/preprocess/join_multiple_reasoning.py --text_reasoning_path $OUTPUT_PATH_TEXT_REASONING \
                                                 --image_reasoning_path $OUTPUT_PATH_IMAGE_REASONING \
                                                 --reasoning_path $OUTPUT_PATH_REASONING \
                                                 --ocr_path $OUTPUT_PATH_OCR \
                                                 --input_path $ANNOTATION_PATH \
                                                 --output_path $OUTPUT_PATH
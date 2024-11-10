ANNOTATION_PATH="data/private_test/vimmsd-private-test.json"
IMAGE_PATH="data/private_test/test-images"
OUTPUT_PATH_OCR="data/private_test/ocr_llm.json"
OUTPUT_PATH_REASONING="data/private_test/reasoning.json"
OUTPUT_PATH_TEXT_REASONING="data/private_test/text_reasoning.json"
OUTPUT_PATH_IMAGE_REASONING="data/private_test/image_reasoning.json"
# prepare ocr using vi_intern
python src/preprocess/extract_text_llm.py --annotation_path $ANNOTATION_PATH \
                                          --image_folder_path $IMAGE_PATH \
                                          --output_path $OUTPUT_PATH_OCR
# prepare reasoning
CUDA_VISIBLE_DEVICES=0 python src/agent/reasoning.py --model "vi_intern" \
                                                     --image_path $IMAGE_PATH \
                                                     --input_path $ANNOTATION_PATH \
                                                     --output_path $OUTPUT_PATH_REASONING \
                                                     --batch_size 1
# prepare text reasoning
python src/agent/text_reasoning.py --model "arcee-ai/Arcee-VyLinh" \
                                   --input_path $ANNOTATION_PATH \
                                   --output_path $OUTPUT_PATH_TEXT_REASONING \
                                   --batch 1
# prepare image reasoning
CUDA_VISIBLE_DEVICES=1 python src/agent/reasoning.py --model "vi_intern_image" \
                                                     --image_path $IMAGE_PATH \
                                                     --input_path $ANNOTATION_PATH \
                                                     --output_path $OUTPUT_PATH_IMAGE_REASONING \
                                                     --batch_size 1
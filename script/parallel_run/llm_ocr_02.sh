CUDA_VISIBLE_DEVICES=1 python src/preprocess/extract_text_llm.py --annotation_path "data/public_train/vimmsd-train_02.json" \
                                                                 --image_folder_path "data/public_train/train-images" \
                                                                 --output_path "data/public_train/ocr_llm_02_v2.json"


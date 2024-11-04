CUDA_VISIBLE_DEVICES=0 python src/agent/reasoning.py --model "vi_intern" \
                                                     --image_path "data/public_train/train-images" \
                                                     --input_path "data/public_train/vimmsd-train_01.json" \
                                                     --output_path "data/public_train/reasoning_vi_intern_image_reasoning_01.json" \
                                                     --batch_size 1
cd craft_pytorch
PYTHONPATH=$(pwd)/craft_pytorch TORCH_HOME=$(pwd)/craft_pytorch python3 test.py --trained_model="../model/craft/craft_mlt_25k.pth" \
                                                              --test_folder="../input_tmp" \
                                                              --bbox_folder="../output_tmp" \
                                                              --cuda True
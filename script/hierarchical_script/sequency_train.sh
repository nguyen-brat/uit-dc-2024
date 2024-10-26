export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_PROJECT="DS-DC"
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:true
#export HF_TOKEN="" put your HF_TOKEN HERE
DIR=`pwd`
DEEPSPEED2_PATH=${DIR}/script/dszero2.yaml
DEEPSPEED3_PATH=${DIR}/script/dszero3.yaml
TRAIN_CONFIG_PATH_1=${DIR}/script/hierarchical_script/cls_not_yet_sarcasm.yaml
TRAIN_CONFIG_PATH_2=${DIR}/script/hierarchical_script/cls_mul_vs_img_text.yaml

# train classify to detect sarcasm or not first
accelerate launch --config_file $DEEPSPEED2_PATH ${DIR}/train.py --config_path $TRAIN_CONFIG_PATH_1

# train classify multi-sarcasm and text + image sarcasm

accelerate launch --config_file $DEEPSPEED2_PATH ${DIR}/train.py --config_path $TRAIN_CONFIG_PATH_2
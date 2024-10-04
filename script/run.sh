export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_PROJECT="DS-DC"
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:true
#export HF_TOKEN="" put your HF_TOKEN HERE
DIR=`pwd`
DEEPSPEED2_PATH=${DIR}/script/dszero2.yaml
DEEPSPEED3_PATH=${DIR}/script/dszero3.yaml
TRAIN_CONFIG_PATH=${DIR}/script/train.yaml

accelerate launch --config_file $DEEPSPEED3_PATH ${DIR}/run.py --config_path $TRAIN_CONFIG_PATH
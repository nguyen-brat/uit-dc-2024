export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_PROJECT="DS-DC"
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:true
#export HF_TOKEN="" put your HF_TOKEN HERE
DIR=`pwd`
DEEPSPEED2_PATH=${DIR}/src/config/deepspeedzero2.yaml
DEEPSPEED3_PATH=${DIR}/src/config/deepspeedzero3.yaml
TRAIN_CONFIG_PATH=${DIR}/src/config/train.yaml

accelerate launch --config_file $DEEPSPEED2_PATH ${DIR}/run.py
# accelerate launch --config_file="src/config/fsdp.yaml" src/train/train.py
# accelerate launch --config_file="src/config/fsdp.yaml" ${DIR}/run.py \
#                     --config_path $TRAIN_CONFIG_PATH
# accelerate launch ${DIR}/run.py
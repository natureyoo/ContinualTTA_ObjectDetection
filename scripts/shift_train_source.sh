#!/bin/bash
TIME=`date +%m%d_%H%M%S`

# swinT
#export config_path="./configs/SHIFT/faster_rcnn_swinT_FPN_1x.yaml"
#export output_dir="./outputs/SHIFT/faster_rcnn_swinT"
# resne50
config_path=./configs/SHIFT/faster_rcnn_R_50_FPN_1x_discrete.yaml
output_dir=./outputs/SHIFT/faster_rcnn_resnet50/${TIME}

# SHIFT config usage: 
#   아래 선택지 중 입력. 현재 복수 선택 불가능. 따로 선택하지 않고자 할 경우 None 입력 혹은 config 선언하지 않음.
#   DATASETS.SHIFT.SHIFT_TYPE = daytime_to_night, clear_to_rainy, clear_to_foggy
#   DATASETS.SHIFT.WEATHER = overcast, clear, cloudy, foggy, rainy
#   DATASETS.SHIFT.TIME = daytime, night, dawn/dusk

# trains source domain
python tools/train_net.py \
    --config-file ${config_path} --num-gpus 2  \
    OUTPUT_DIR ${output_dir} \
    SOLVER.IMS_PER_BATCH 16 \
    SOLVER.IMS_PER_BATCH_TEST 4 \
    DATASETS.SHIFT.SHIFT_TYPE None  \
    DATASETS.SHIFT.WEATHER clear \
    DATASETS.SHIFT.TIME None 


# collect features
# python tools/train_net.py \
#   	--config-file ${config_path} \
#   	--eval-only \
#   	MODEL.WEIGHTS ${output_dir}/model_final.pth \
#   	SOLVER.IMS_PER_BATCH_TEST 16 \
#   	TEST.COLLECT_FEATURES "True" \
# 	  OUTPUT_DIR ${output_dir}

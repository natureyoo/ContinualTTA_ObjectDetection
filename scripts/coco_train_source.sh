#!/bin/bash
TIME=`date +%m%d_%H%M%S`

# swinT
#export config_path="./configs/COCO-Detection/faster_rcnn_swinT_FPN_1x.yaml"
#export output_dir="./outputs/COCO/faster_rcnn_swinT"
# resne50
config_path=./configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml
output_dir=./outputs/COCO/faster_rcnn_resnet50/${TIME}

# trains source domain
nohup python tools/train_net.py \
    --config-file ${config_path} \
    OUTPUT_DIR ${output_dir} \
	SOLVER.IMS_PER_BATCH 8 \
	SOLVER.IMS_PER_BATCH_TEST 4 \
	TEST.EVAL_PERIOD 5000 
	

# collect features
# python tools/train_net.py \
#   	--config-file ${config_path} \
#   	--eval-only \
#   	MODEL.WEIGHTS ${output_dir}/model_final.pth \
#   	SOLVER.IMS_PER_BATCH_TEST 16 \
#   	TEST.COLLECT_FEATURES "True" \
# 	  OUTPUT_DIR ${output_dir}
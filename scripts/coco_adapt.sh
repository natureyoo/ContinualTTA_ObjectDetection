#!/bin/bash

# swinT
#export config_path1="./configs/COCO-Detection/faster_rcnn_swinT_FPN_1x.yaml"
#export config_path2="./configs/TTA/COCO_swinT.yaml"
#export output_dir="./outputs/COCO/faster_rcnn_swinT"
#export feature_path="./models/faster_rcnn_swinT_coco_feature_stats.pt"
# resne50
export config_path1="./configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
export config_path2="./configs/TTA/COCO_resnet50.yaml"
export output_dir="./outputs/COCO/faster_rcnn_resnet50"
export feature_path="./models/faster_rcnn_r50_coco_feature_stats.pt"

# Direct Test
python tools/train_net.py \
  	--config-file ${config_path1} \
  	--eval-only --wandb \
  	MODEL.WEIGHTS ${output_dir}/model_final.pth \
  	SOLVER.IMS_PER_BATCH_TEST 16 \
  	TEST.CONTINUAL_DOMAIN "True" \
	  OUTPUT_DIR ${output_dir}_direct_test

# full finetuning
export where="full"
for continual in False True
do python tools/train_net.py \
  	--config-file ${config_path2} \
  	--eval-only --wandb \
  	TEST.ADAPTATION.WHERE ${where} \
  	TEST.ADAPTATION.CONTINUAL ${continual} \
	  OUTPUT_DIR ${output_dir}_${where}_continual_${continual}
done

# adapter tuning
export where="adapter"
for continual in False True
do python tools/train_net.py \
  	--config-file ${config_path2} \
  	--eval-only --wandb \
  	SOLVER.BASE_LR 0.001 \
  	TEST.ADAPTATION.WHERE ${where} \
  	TEST.ADAPTATION.CONTINUAL ${continual} \
	  OUTPUT_DIR ${output_dir}_${where}_continual_${continual}
done
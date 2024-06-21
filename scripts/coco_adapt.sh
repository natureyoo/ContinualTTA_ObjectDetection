#!/bin/bash

# Backbone Type
export bb="R50" #"swinT"

# Set config file
export config_path="configs/TTA/COCO_${bb}.yaml"

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
  	--config-file "./configs/COCO-Detection/faster_rcnn_${bb}_FPN_1x.yaml" \
  	--eval-only --wandb \
  	TEST.CONTINUAL_DOMAIN "True" \
	OUTPUT_DIR outputs/COCO/${bb}_direct_test

# full finetuning
export where="full"
for continual in False True
do python tools/train_net.py \
  	--config-file ${config_path} \
  	--eval-only --wandb \
  	TEST.ADAPTATION.WHERE ${where} \
  	TEST.ADAPTATION.CONTINUAL ${continual} \
	OUTPUT_DIR outputs/COCO/${bb}_full_finetuning_continual_${continual}
done

# ours
export where="adapter"
for continual in False True
do python tools/train_net.py \
  	--config-file ${config_path} \
  	--eval-only --wandb \
  	TEST.ADAPTATION.WHERE ${where} \
  	TEST.ADAPTATION.CONTINUAL ${continual} \
	OUTPUT_DIR outputs/COCO/${bb}_ours_continual_${continual}
done

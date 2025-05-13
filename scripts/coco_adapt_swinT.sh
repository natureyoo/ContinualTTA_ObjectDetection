#!/bin/bash

# Backbone Type
export bb="swinT" #"swinT"

# Set config file
export config_path="configs/TTA/COCO_${bb}.yaml"

# Collect Feature Statistics
python tools/train_net.py \
  	--config-file "./configs/Base/COCO_faster_rcnn_${bb}_FPN_1x.yaml" \
  	--eval-only --wandb \
	MODEL.WEIGHTS models/checkpoints/faster_rcnn_${bb}_coco.pth \
  	TEST.CONTINUAL_DOMAIN "False" \
	TEST.COLLECT_FEATURES "True" \
	OUTPUT_DIR outputs/COCO/${bb}_collect_feature_stats

# Direct Test
python tools/train_net.py \
  	--config-file "./configs/Base/COCO_faster_rcnn_${bb}_FPN_1x.yaml" \
  	--eval-only --wandb \
	MODEL.WEIGHTS models/checkpoints/faster_rcnn_${bb}_coco.pth \
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
for continual in True
do
	python tools/train_net.py \
  	--config-file ${config_path} \
  	--eval-only --wandb \
  	TEST.ADAPTATION.WHERE ${where} \
  	TEST.ADAPTATION.CONTINUAL ${continual} \
	TEST.ADAPTATION.SOURCE_FEATS_PATH "./models/stats/COCO_${bb}_stats.pt" \
	OUTPUT_DIR outputs/COCO/${bb}_ours_continual_${continual}
done

# ours-skip
export where="adapter"
for continual in True
do 
	python tools/train_net.py \
  	--config-file ${config_path} \
  	--eval-only --wandb \
  	TEST.ADAPTATION.WHERE ${where} \
  	TEST.ADAPTATION.CONTINUAL ${continual} \
    TEST.ADAPTATION.SKIP_REDUNDANT "stat-period-ema" \
	TEST.ADAPTATION.SOURCE_FEATS_PATH "./models/stats/COCO_${bb}_stats.pt" \
	OUTPUT_DIR outputs/COCO/${bb}_ours_skip_continual_${continual}
done

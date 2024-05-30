#!/bin/bash

# Mean-Teacher

# SHIFT Discrete ResNet50
python tools/train_net.py \
	--config-file configs/TTA/SHIFT_R50.yaml \
	--eval-only --wandb \
	SOLVER.BASE_LR 0.0001 \
	SOLVER.CLIP_GRADIENTS.CLIP_VALUE 1.0 \
	TEST.ADAPTATION.WHERE "full" \
	TEST.ADAPTER.TYPE "None" \
	TEST.ADAPTATION.TYPE mean-teacher \
	OUTPUT_DIR outputs/SHIFT_Discrete_CTA/R50_mean-teacher_lr0.0001_clip1.0

# SHIFT Continuous100x ResNet50
export dir_name="continuous100x"
for shift_type in "clear_to_foggy" "clear_to_rainy"
  do python tools/train_net.py \
	--config-file configs/TTA/SHIFT_R50.yaml \
	--eval-only --wandb \
        DATASETS.SHIFT.SHIFT_TYPE ${shift_type} \
	DATASETS.SHIFT.DIRNAME ${dir_name} \
	SOLVER.BASE_LR 0.0001 \
	SOLVER.CLIP_GRADIENTS.CLIP_VALUE 1.0 \
	TEST.ADAPTATION.WHERE "full" \
	TEST.ADAPTER.TYPE "None" \
	TEST.ADAPTATION.TYPE mean-teacher \
	OUTPUT_DIR outputs/SHIFT_${dir_name}_${shift_type}/R50_mean-teacher_lr0.0001_clip1.0
  done

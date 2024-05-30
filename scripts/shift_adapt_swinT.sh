#!/bin/bash

export config_path="configs/TTA/SHIFT_discrete_swinT.yaml"

# full tuning
for lr in 0.0001 0.00001
   do python tools/train_net.py \
      --config-file ${config_path} \
      --eval-only --wandb \
      SOLVER.BASE_LR ${lr} \
      SOLVER.CLIP_GRADIENTS.CLIP_VALUE 1.0 \
      TEST.ADAPTATION.WHERE "full" \
      TEST.ADAPTER.TYPE "None" \
      OUTPUT_DIR outputs/SHIFT_Discrete_CTA/swinT_full_lr${lr}_clip1.0
   done

# normalization tuning
for lr in 0.0001 0.00001
do python tools/train_net.py \
      --config-file ${config_path} \
      --eval-only --wandb \
      SOLVER.BASE_LR ${lr} \
      SOLVER.CLIP_GRADIENTS.CLIP_VALUE 1.0 \
      TEST.ADAPTATION.WHERE "normalization" \
      TEST.ADAPTER.TYPE "None" \
      OUTPUT_DIR outputs/SHIFT_Discrete_CTA/swinT_LN_lr${lr}_clip1.0
done

# roi head tuning
for lr in 0.0001 0.00001
do python tools/train_net.py \
      --config-file ${config_path} \
      --eval-only --wandb \
      SOLVER.BASE_LR ${lr} \
      SOLVER.CLIP_GRADIENTS.CLIP_VALUE 1.0 \
      TEST.ADAPTATION.WHERE "head" \
      TEST.ADAPTER.TYPE "None" \
      OUTPUT_DIR outputs/SHIFT_Discrete_CTA/swinT_head_lr${lr}_clip1.0
done


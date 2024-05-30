#!/bin/bash

export config_path="configs/TTA/SHIFT_discrete_R50.yaml"

# direct test
 python tools/train_net.py \
      --config-file ${config_path} \
      --eval-only --wandb \
      TEST.ONLINE_ADAPTATION False \
      OUTPUT_DIR outputs/SHIFT_Discrete_CTA/R50_direct_test

# full tuning
for lr in 0.0001
   do python tools/train_net.py \
      --config-file ${config_path} \
      --eval-only --wandb \
      SOLVER.BASE_LR ${lr} \
      SOLVER.CLIP_GRADIENTS.CLIP_VALUE 1.0 \
      TEST.ADAPTATION.WHERE "full" \
      TEST.ADAPTER.TYPE "None" \
      OUTPUT_DIR outputs/SHIFT_Discrete_CTA/R50_full_lr${lr}_clip1.0
   done

# adapter tuning
for clip in 1.0 3.0 5.0
   do python tools/train_net.py \
      --config-file ${config_path} \
      --eval-only --wandb \
      SOLVER.BASE_LR 0.001 \
      SOLVER.CLIP_GRADIENTS.CLIP_VALUE ${clip} \
      TEST.ADAPTATION.WHERE "adapter" \
      TEST.ADAPTER.TYPE "parallel" \
      OUTPUT_DIR outputs/SHIFT_Discrete_CTA/R50_adapter_lr0.001_clip${clip}
   done

# normalization tuning
for lr in 0.0001 0.00001 0.001
do python tools/train_net.py \
      --config-file ${config_path} \
      --eval-only --wandb \
      SOLVER.BASE_LR ${lr} \
      SOLVER.CLIP_GRADIENTS.CLIP_VALUE 1.0 \
      TEST.ADAPTATION.WHERE "normalization" \
      TEST.ADAPTER.TYPE "None" \
      OUTPUT_DIR outputs/SHIFT_Discrete_CTA/R50_BN_lr${lr}_clip1.0
done

# roi head tuning
for lr in 0.0001 0.00001 0.001
do python tools/train_net.py \
      --config-file ${config_path} \
      --eval-only --wandb \
      SOLVER.BASE_LR ${lr} \
      SOLVER.CLIP_GRADIENTS.CLIP_VALUE 1.0 \
      TEST.ADAPTATION.WHERE "head" \
      TEST.ADAPTER.TYPE "None" \
      OUTPUT_DIR outputs/SHIFT_Discrete_CTA/R50_head_lr${lr}_clip1.0
done


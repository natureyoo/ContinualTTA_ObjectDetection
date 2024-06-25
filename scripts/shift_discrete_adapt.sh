#!/bin/bash

# Backbone Type
export bb="R50" #"swinT"

# Set config file
export config_path="configs/TTA/SHIFT_discrete_${bb}.yaml"

# direct test
python tools/train_net.py \
      --config-file ${config_path} \
      --eval-only --wandb \
      TEST.ONLINE_ADAPTATION False \
      OUTPUT_DIR outputs/SHIFT_Discrete_CTA/${bb}_direct_test

# full tuning
python tools/train_net.py \
      --config-file ${config_path} \
      --eval-only --wandb \
      TEST.ADAPTATION.WHERE "full" \
      TEST.ADAPTER.TYPE "None" \
      OUTPUT_DIR outputs/SHIFT_Discrete_CTA/${bb}_full_finetuning

# ours
python tools/train_net.py \
      --config-file ${config_path} \
      --eval-only --wandb \
      TEST.ADAPTATION.WHERE "adapter" \
      TEST.ADAPTER.TYPE "parallel" \
      OUTPUT_DIR outputs/SHIFT_Discrete_CTA/${bb}_ours


#!/bin/bash


for shift_type in "clear_to_foggy" "clear_to_rainy"
    do for dir_name in "continuous100x" "continuous10x"
            do python tools/train_net.py \
                  --config-file "./configs/TTA/SHIFT_swinT.yaml" \
                  --eval-only --wandb \
                  DATASETS.SHIFT.SHIFT_TYPE ${shift_type} \
                  DATASETS.SHIFT.DIRNAME ${dir_name} \
                  TEST.ADAPTATION.WHERE "full" \
                  OUTPUT_DIR outputs/SHIFT_${dir_name}_${shift_type}_TTA/swinT_clear_daytime_source_full_update
        done
    done

for shift_type in "clear_to_foggy" "clear_to_rainy"
    do for dir_name in "continuous100x" "continuous10x"
        do for lr in 0.01 0.001 0.0001
            do python tools/train_net.py \
                  --config-file "./configs/TTA/SHIFT_swinT.yaml" \
                  --eval-only --wandb \
                  DATASETS.SHIFT.SHIFT_TYPE ${shift_type} \
                  DATASETS.SHIFT.DIRNAME ${dir_name} \
                  SOLVER.BASE_LR ${lr} \
                  TEST.ADAPTATION.WHERE "adapter" \
                  OUTPUT_DIR outputs/SHIFT_${dir_name}_${shift_type}_TTA/swinT_clear_daytime_source_adapter_update_lr_${lr}
            done
        done
    done
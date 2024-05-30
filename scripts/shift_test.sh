#!/bin/bash

for shift_type in "clear_to_foggy" "clear_to_rainy"
    do for dir_name in "continuous10x"
            do python tools/train_net.py \
                  --config-file "./configs/SHIFT/faster_rcnn_R_50_FPN_1x_discrete.yaml" \
                  --eval-only --wandb \
                  MODEL.WEIGHTS "models/faster_rcnn_R50_shift_clear_daytime.pth" \
                  DATASETS.SHIFT.SHIFT_TYPE ${shift_type} \
                  DATASETS.SHIFT.DIRNAME ${dir_name} \
                  TEST.CONTINUAL_DOMAIN True \
                  OUTPUT_DIR outputs/SHIFT_${dir_name}_${shift_type}_direct_test/R50_clear_daytime_source
        done
    done

for shift_type in "clear_to_foggy" "clear_to_rainy"
    do for dir_name in "continuous10x"
            do python tools/train_net.py \
                  --config-file "./configs/SHIFT/faster_rcnn_swinT_FPN_1x_discrete.yaml" \
                  --eval-only --wandb \
                  MODEL.WEIGHTS "models/faster_rcnn_swinT_shift_clear_daytime.pth" \
                  DATASETS.SHIFT.SHIFT_TYPE ${shift_type} \
                  DATASETS.SHIFT.DIRNAME ${dir_name} \
                  TEST.CONTINUAL_DOMAIN True \
                  OUTPUT_DIR outputs/SHIFT_${dir_name}_${shift_type}_direct_test/swinT_clear_daytime_source
        done
    done

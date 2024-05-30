#!/bin/bash

# Mean-Teacher
#python tools/train_net.py \
#                --config-file ./configs/TTA/COCO_resnet50.yaml \
#                --eval-only --wandb \
#                SOLVER.BASE_LR 0.0001 \
#                SOLVER.CLIP_GRADIENTS.CLIP_VALUE 1.0 \
#                TEST.ADAPTATION.WHERE full \
#		TEST.ADAPTER.TYPE None \
#                TEST.ADAPTATION.TYPE mean-teacher \
#                OUTPUT_DIR outputs/COCO_CTA/R50_mean-teacher_lr0.0001_clip1.0

python tools/train_net.py \
                --config-file ./configs/TTA/COCO_swinT.yaml \
                --eval-only --wandb \
                SOLVER.BASE_LR 0.0001 \
                SOLVER.CLIP_GRADIENTS.CLIP_VALUE 1.0 \
                TEST.ADAPTATION.WHERE full \
		TEST.ADAPTER.TYPE None \
                TEST.ADAPTATION.TYPE mean-teacher \
		TEST.ADAPTATION.EMA_BETA 0.999 \
                OUTPUT_DIR outputs/COCO_CTA/swinT_mean-teacher_lr0.0001_clip1.0_beta0.999

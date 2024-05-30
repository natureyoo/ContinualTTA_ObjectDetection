#!/bin/bash

for shift_type in "clear_to_foggy" "clear_to_rainy"
	do for dir_name in "continuous100x" "continuous10x"
		do python tools/train_net.py \
		--config-file "./configs/SHIFT/faster_rcnn_swinT_FPN_1x_discrete.yaml" --eval-only --wandb \
		MODEL.WEIGHTS models/faster_rcnn_swinT_shift_clear_daytime.pth \
		TEST.CONTINUAL_DOMAIN True \
		DATASETS.SHIFT.SHIFT_TYPE ${shift_type} \
		DATASETS.SHIFT.DIRNAME ${dir_name} \
		OUTPUT_DIR outputs/SHIFT/swinT_clear_daytime_source_direct_test_${dir_name}_${shift_type}
	done
done

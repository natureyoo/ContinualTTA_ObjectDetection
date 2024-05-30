#!/bin/bash

for lr in 0.001 0.005
do for shift_type in "clear_to_foggy" "clear_to_rainy"
	do for dir_name in "continuous100x"
		do python tools/train_net.py \
		--config-file configs/TTA/SHIFT_R50.yaml --eval-only --wandb \
		DATASETS.SHIFT.SHIFT_TYPE ${shift_type} \
		DATASETS.SHIFT.DIRNAME ${dir_name} \
		SOLVER.BASE_LR ${lr} \
		SOLVER.CLIP_GRADIENTS.CLIP_VALUE 5.0 \
		TEST.ONLINE_ADAPTATION True \
		TEST.ADAPTATION.WHERE adapter \
		TEST.ADAPTER.TYPE parallel \
		OUTPUT_DIR outputs/SHIFT_${dir_name}_${shift_type}/R50_adapter_lr${lr}_clip5.0
        done
	done
done

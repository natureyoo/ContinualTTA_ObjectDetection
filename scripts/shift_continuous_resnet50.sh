#!/bin/bash

# Backbone Type
export bb="R50" #"swinT"

# Set config file
export config_path="configs/TTA/SHIFT_continuous_${bb}.yaml"

# CTA via ours
for shift_type in "clear_to_foggy" "clear_to_rainy"
	do for dir_name in "continuous100x" "continuous10x"
		do python tools/train_net.py \
		--config-file configs/TTA/SHIFT_continuous_R50.yaml --eval-only --wandb \
		DATASETS.SHIFT.SHIFT_TYPE ${shift_type} \
		DATASETS.SHIFT.DIRNAME ${dir_name} \
		TEST.ONLINE_ADAPTATION True \
		TEST.ADAPTATION.WHERE adapter \
		TEST.ADAPTER.TYPE parallel \
		OUTPUT_DIR outputs/SHIFT_${dir_name}_${shift_type}/${bb}_ours
        done
	done

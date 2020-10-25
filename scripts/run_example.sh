#!/bin/bash

data_dir="/home/cv5/moon/Gated2Depth/example"
results_dir="/home/cv5/data/gated2depth/results/example"
gpu=0

model_dir="/home/cv5/data/gated2depth/gated2depth/models/gated2depth_real_night/model.ckpt-8028"

python /home/cv5/moon/Gated2Depth/src/train_eval.py \
    --results_dir $results_dir/night \
	  --model_dir $model_dir \
	  --eval_files_path /home/cv5/moon/Gated2Depth/splits/example_night.txt \
	  --base_dir $data_dir \
	  --data_type real \
	  --gpu $gpu \
	  --mode eval \
	  --show_result

model_dir="/home/cv5/data/gated2depth/gated2depth/models/gated2depth_real_day/model.ckpt-13460"

python /home/cv5/moon/Gated2Depth/src/train_eval.py \
    --results_dir $results_dir/day \
	  --model_dir $model_dir \
	  --eval_files_path /home/cv5/moon/Gated2Depth/splits/example_day.txt \
	  --base_dir $data_dir \
	  --data_type real \
	  --gpu $gpu \
	  --mode eval \
	  --show_result


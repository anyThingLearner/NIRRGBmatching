#!/bin/bash

split="test" # or "val"
data_dir="/home/cv5/data/gated2depth/gated2depth/real"
model_dir="/home/cv5/data/gated2depth/gated2depth/models/gated2depth_real_night/model.ckpt-8028"
results_dir="/home/cv5/data/gated2depth/results/gated2depth_real_night/${split}"
eval_files="/home/cv5/moon/Gated2Depth/splits/real_${split}_night.txt"
gpu=0

python /home/cv5/moon/Gated2Depth/src/train_eval.py \
    --results_dir $results_dir \
	  --model_dir $model_dir \
	  --eval_files_path $eval_files \
	  --base_dir $data_dir \
	  --data_type real \
	  --gpu $gpu \
	  --mode eval


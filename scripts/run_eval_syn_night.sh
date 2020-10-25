#!/bin/bash

split="test" # or "val"
data_dir="/home/cv5/data/gated2depth/gated2depth/syn"
model_dir="/home/cv5/data/gated2depth/gated2depth/models/gated2depth_syn_night/model.ckpt-38850"
results_dir="/home/cv5/data/gated2depth/results/gated2depth_syn_night_test/${split}"
eval_files="/home/cv5/moon/my_ex/Gated2Depth/splits/syn_${split}_night.txt"
gpu=0

python /home/cv5/moon/my_ex/Gated2Depth/src/train_eval.py \
    --results_dir $results_dir \
	  --model_dir $model_dir \
	  --eval_files_path $eval_files \
	  --base_dir $data_dir \
	  --data_type synthetic \
	  --gpu $gpu \
	  --mode eval


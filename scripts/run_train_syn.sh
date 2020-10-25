#!/bin/bash

daytime="night" # or "day"
data_dir="/home/cv5/data/gated2depth/gated2depth/syn"
#model_dir="/home/cv5/data/gated2depth/gated2depth/models/gated2depth_real_night/model.ckpt-8028"
model_dir="/home/cv5/data/gated2depth/gated2depth/models/model_syn_${daytime}"
results_dir="/home/cv5/data/gated2depth/results/syn_${daytime}_test"
train_files="/home/cv5/moon/my_ex/Gated2Depth/splits/syn_train_${daytime}.txt"
eval_files="/home/cv5/moon/my_ex/Gated2Depth/splits/syn_val_${daytime}.txt"
gpu=0

python /home/cv5/moon/my_ex/Gated2Depth/src/train_eval.py \
    --results_dir $results_dir \
    --model_dir $model_dir \
    --train_files_path $train_files \
    --eval_files_path $eval_files \
    --base_dir $data_dir \
    --data_type synthetic \
    --gpu $gpu \
    --mode train \
    --num_epochs 2 \
    --lrate 0.0001 \
    --smooth_weight 0.5 \
##    --use_3dconv 3d


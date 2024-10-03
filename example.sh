#!/bin/bash

# Activate your python environment, if you're using one
# conda activate vot

torchrun --nproc_per_node 2 train.py \
--vit_architecture 'swin' \
--learning_rate 0.0001 \
--batch_size 4 \
--epochs 100 \
--data_paths '/proj/cloudrobotics-nest/users/x_shuji/vot_test/20230815_1/robotCR19' \
--test_paths '/proj/cloudrobotics-nest/users/x_shuji/vot_test/20230815_1/robotCR19/Test' \
--ckpt_path '/proj/cloudrobotics-nest/users/x_shuji/OccluManip_models/test' 
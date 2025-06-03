#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

lambda_NR=0.5
lambda_CC=0.5
inject_layers=11,12,13,14,15,16,17,18,19,20
name=exp_nr_"$lambda_NR"_cc_"$lambda_CC"_embeds

python train.py --dataroot /share/nas169/jethrowang/URSA-GAN/data/UNA-GAN --name "$name" --lambda_NR "$lambda_NR" --lambda_CC "$lambda_CC" --inject_layers "$inject_layers" --CUT_mode CUT --checkpoints_dir checkpoints

for i in {1..10}; do
    python test.py --dataroot /share/nas169/jethrowang/URSA-GAN/data/UNA-GAN --name "$name" --source_idx "$i" --lambda_NR "$lambda_NR" --lambda_CC "$lambda_CC" --inject_layers "$inject_layers" --CUT_mode CUT --checkpoints_dir checkpoints_"$language" --state Test --results_dir ./results
done

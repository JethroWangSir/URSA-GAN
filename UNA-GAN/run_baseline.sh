#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

name=exp_baseline

python train.py --dataroot /share/nas169/jethrowang/URSA-GAN/data/UNA-GAN --name "$name" --CUT_mode CUT --checkpoints_dir checkpoints

for i in {1..10}; do
    python test.py --dataroot /share/nas169/jethrowang/URSA-GAN/data/UNA-GAN --name "$name" --source_idx "$i" --CUT_mode CUT --checkpoints_dir checkpoints_"$language" --state Test --results_dir ./results
done

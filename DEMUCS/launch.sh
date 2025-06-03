#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

export CUDA_VISIBLE_DEVICES=2

python train.py \
  continue_pretrained=master64 \
  dset=hat \
  segment=4.5 \
  stride=0.5 \
  remix=1 \
  bandmask=0.2 \
  shift=8000 \
  shift_same=True \
  lr=3e-4 \
  stft_loss=True \
  stft_sc_factor=0.1 stft_mag_factor=0.1 \
  epochs=10 \
  batch_size=16 \
  demucs.causal=1 \
  demucs.hidden=64 \
  ddp=1 \
  hydra.run.dir=./outputs/exp_topline $@

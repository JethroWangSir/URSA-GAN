#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

export CUDA_VISIBLE_DEVICES=0

path_tr=egs/hat-esc_nr_0.5_cc_0.5_embeds/tr
if [[ ! -e $path_tr ]]; then
    mkdir -p $path_tr
fi
noisy_train=/share/nas169/jethrowang/URSA-GAN/UNA-GAN/results/exp_nr_0.5_cc_0.5_embeds/test_latest/audios/fake_B
clean_train=/share/nas169/jethrowang/URSA-GAN/data/HAT/train/condenser

python -m denoiser.audio $noisy_train > $path_tr/noisy.json
python -m denoiser.audio $clean_train > $path_tr/clean.json

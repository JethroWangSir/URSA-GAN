#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

export CUDA_VISIBLE_DEVICES=0

path_tt=egs/hat-esc/tt
if [[ ! -e $path_tt ]]; then
    mkdir -p $path_tt
fi
noisy_test=/share/nas169/jethrowang/URSA-GAN/data/HAT-ESC/test_webcam+target_noise/all
clean_test=/share/nas169/jethrowang/URSA-GAN/data/HAT/test/condenser

python -m denoiser.audio $noisy_test > $path_tt/noisy.json
python -m denoiser.audio $clean_test > $path_tt/clean.json
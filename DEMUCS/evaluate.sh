export CUDA_VISIBLE_DEVICES=0

# python -m denoiser.evaluate --model_path=<path to the model> --data_dir=<path to folder containing noisy.json and clean.json>
python -m denoiser.evaluate --master64 --data_dir=/share/nas169/jethrowang/URSA-GAN/DEMUCS/egs/hat-esc/tt

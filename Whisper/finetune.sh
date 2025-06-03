export CUDA_VISIBLE_DEVICES=0

TRAIN_BATCH_SIZE=64
TEST_BATCH_SIZE=32
NUM_EPOCHS=30
LEARNING_RATE=5e-4

DATASET='hat'
LANGUAGE='hakka'
EXP_NAME='exp_nr_0.5_cc_0.5_embeds'
GENERATED_TRAIN_AUDIO_DIR="/share/nas169/jethrowang/URSA-GAN/UNA-GAN/results/$EXP_NAME/test_latest/audios/fake_B"
TOPLINE_TRAIN_AUDIO_DIR='/share/nas169/jethrowang/URSA-GAN/data/HAT-ESC/train_webcam+target_noise'
TEST_AUDIO_DIR='/share/nas169/jethrowang/URSA-GAN/data/HAT-ESC/test_webcam+target_noise/all'
OUTPUT_DIR="./results/whisper-tiny_hat-esc_$EXP_NAME"

python finetune.py \
  --dataset $DATASET \
  --language $LANGUAGE \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --test_batch_size $TEST_BATCH_SIZE \
  --num_epochs $NUM_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --exp_name $EXP_NAME \
  --generated_train_audio_dir $GENERATED_TRAIN_AUDIO_DIR \
  --topline_train_audio_dir $TOPLINE_TRAIN_AUDIO_DIR \
  --test_audio_dir $TEST_AUDIO_DIR \
  --output_dir $OUTPUT_DIR
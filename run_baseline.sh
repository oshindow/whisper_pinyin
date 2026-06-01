#!/usr/bin/env bash

DATA_ROOT="/path/to/data/"
EXP_DIR="/path/to/exp/"
 

CUDA_VISIBLE_DEVICES=1 python scripts/baseline/finetuning_pinyin_otc.py \
  --epoch 10 \
  --data-root $DATA_ROOT \
  --train-name whisper_pinyin_aishell3_otc \
  --train-id 001 \
  --exp-dir $EXP_DIR \
  --train-path dump/aishell3/train/text \
  --model-name small \
  --ctc-layers 2 \
  --n_mels 80 \
  --batch-size 32 \
  --precision bf16-mixed \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --adam-epsilon 1e-8 \
  --warmup-steps 1000 > exp_whisper_otc.log
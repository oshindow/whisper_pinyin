#!/usr/bin/env bash

CHECKPOINT="/path/to/checkpoint"
TEST_PATH="dump/aishell3/test/text"
OUTPUT="results_otc_aishell3_test.txt"
DATA_ROOT="/path/to/data/"

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python inference/inference_pinyin_ctc.py \
  --checkpoint $CHECKPOINT \
  --test-path $TEST_PATH \
  --data-root $DATA_ROOT \
  --batch-size 128 \
  --output $OUTPUT 

#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
CHECKPOINT="${CHECKPOINT:-/path/to/checkpoint.ckpt}"
TEST_PATH="${TEST_PATH:-dump/aishell3/test/text}"
OUTPUT="${OUTPUT:-results_aishell3_test.txt}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python inference/inference_pinyin_ctc.py \
  --checkpoint "$CHECKPOINT" \
  --test-path "$TEST_PATH" \
  --output "$OUTPUT"

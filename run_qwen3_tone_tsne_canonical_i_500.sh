#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/xintong/miniconda3/envs/qwen3_omni/bin/python
CHECKPOINT=/data2/xintong/checkpoints/qwen3_asr_ctc_actual_twostage/checkpoints/checkpoint-step=030000-per=0.0375.ckpt

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/tmp/qwen3-matplotlib-cache "$PYTHON" \
  scripts/analyze_qwen3_tone_latents.py \
  --checkpoint "$CHECKPOINT" \
  --manifest data/jsonl_actual_split/train.jsonl \
  --data-root /data2/xintong/datasets/datasets \
  --audio-field canonical_wav \
  --audio-root /data2/xintong/datasets/cosyvoice3/cosyvoice3 \
  --phone-field canonical_phones \
  --final i \
  --samples-per-tone 500 \
  --candidate-multiplier 2 \
  --batch-size 16 \
  --num-workers 0 \
  --seed 42 \
  --output-dir results/qwen3_asr_ctc_step030000/tone_tsne_canonical_i_500_per_tone

#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/xintong/miniconda3/envs/joycent/bin/python
CHECKPOINT=/data2/xintong/checkpoints/whisper-pinyin-ctc/whisper_pinyin_ctc_medium_paired/001/checkpoint-step=030000-per=4.12.ckpt
COMMON=(--checkpoint "$CHECKPOINT" --manifest data/jsonl_actual_paired_split/train.jsonl
  --final i --samples-per-tone 500 --candidate-multiplier 2 --batch-size 1
  --num-workers 0 --seed 42)

MPLCONFIGDIR=/tmp/whisper-paired-matplotlib "$PYTHON" scripts/analyze_whisper_paired_tone_latents.py \
  "${COMMON[@]}" --audio-field l2_wav --phone-field actual_phones \
  --output-dir results/whisper_pinyin_ctc_medium_paired_step030000/tone_tsne_l2_i_500_per_tone

MPLCONFIGDIR=/tmp/whisper-paired-matplotlib "$PYTHON" scripts/analyze_whisper_paired_tone_latents.py \
  "${COMMON[@]}" --audio-field canonical_wav --phone-field canonical_phones \
  --audio-root /data2/xintong/datasets/cosyvoice3/cosyvoice3 \
  --output-dir results/whisper_pinyin_ctc_medium_paired_step030000/tone_tsne_canonical_i_500_per_tone

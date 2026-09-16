#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/.venv-qwen3-ctc/bin/python}"
CHECKPOINT="${CHECKPOINT:-/data2/xintong/checkpoints/qwen3_asr_ctc_actual_tonecontrast_finalbucket_acoustic/checkpoints/last.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-results/qwen3_finalbucket_step29007/tone_tsne_l2_top5_finals_200_per_tone}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.cache-qwen3/huggingface}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/qwen3-bucket-matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

exec "$PYTHON_BIN" scripts/analyze_qwen3_tone_latents.py \
  --checkpoint "$CHECKPOINT" \
  --manifest data/jsonl_actual_split/train.jsonl \
  --data-root /data2/xintong/mandarin_accent \
  --audio-field l2_wav --phone-field actual_phones \
  --num-finals 5 --samples-per-tone 200 --candidate-multiplier 2 \
  --batch-size 16 --num-workers 0 --seed 42 \
  --output-dir "$OUTPUT_DIR" "$@"

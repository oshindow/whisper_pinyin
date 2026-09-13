#!/usr/bin/env bash
set -euo pipefail

# Qwen3-ASR requires a recent Transformers/Python stack; keep the old Whisper
# environment untouched because it is pinned to transformers==4.46.3.
PYTHON_BIN="${PYTHON_BIN:-/home/xintong/miniconda3/envs/qwen3_omni/bin/python}"
DATA_ROOT="/data2/xintong/datasets/datasets"
SPLIT_DIR="data/jsonl_actual_split"
EXP_DIR="/data2/xintong/checkpoints/qwen3_asr_ctc_actual_twostage"
export PYTHONUNBUFFERED=1

if [[ "${FORCE_SPLIT:-0}" == "1" || ! -s "$SPLIT_DIR/train.jsonl" || ! -s "$SPLIT_DIR/dev.jsonl" ]]; then
  echo "[stage 1/2] Scanning audio durations and creating the stratified split..."
  "$PYTHON_BIN" scripts/prepare_jsonl_ctc_splits.py \
    --input train.jsonl --data-root "$DATA_ROOT" --output-dir "$SPLIT_DIR" \
    --dev-ratio 0.03 --min-duration 0.3 --max-duration 20.0 --seed 42
else
  echo "[stage 1/2] Reusing existing split in $SPLIT_DIR (set FORCE_SPLIT=1 to rebuild)."
fi

echo "[stage 2/2] Loading Qwen3-ASR audio-encoder weights and starting DDP training..."
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
"$PYTHON_BIN" scripts/baseline/finetuning_pinyin_ctc_qwen3.py \
  --data-root "$DATA_ROOT" \
  --train-path "$SPLIT_DIR/train.jsonl" \
  --val-path "$SPLIT_DIR/dev.jsonl" \
  --token-table-path data/lang_jsonl/tokens.txt \
  --model-name Qwen/Qwen3-ASR-0.6B-hf \
  --ctc-vocab 188 \
  --exp-dir "$EXP_DIR" \
  --batch-size 4 \
  --gradient-accumulation-steps 3 \
  --precision 16-mixed \
  --learning-rate 5e-5 \
  --adam-epsilon 1e-8 \
  --max-steps 30000 \
  --freeze-backbone-steps 10000 \
  --backbone-ramp-steps 3000 \
  --head-warmup-steps 3000 \
  --eval-steps 1000 \
  --mask-time-prob 0.65 \
  --mask-time-length 10 \
  --mask-feature-prob 0.25 \
  --mask-feature-length 64 \
  --seed 42 \
  --devices 4 \
  --strategy ddp

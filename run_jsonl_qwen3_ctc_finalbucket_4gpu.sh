#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/.venv-qwen3-ctc/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data2/xintong/mandarin_accent}"
SPLIT_DIR="${SPLIT_DIR:-data/jsonl_actual_split}"
EXP_DIR="${EXP_DIR:-/data2/xintong/checkpoints/qwen3_asr_ctc_actual_tonecontrast_finalbucket_acoustic}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.cache-qwen3/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# Instance candidates come from the current per-GPU microbatch, not accumulation.
export BATCH_SIZE="${BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-3}"
export TONE_START_STEP="${TONE_START_STEP:-10000}"
export TONE_RAMP_STEPS="${TONE_RAMP_STEPS:-2000}"
export TONE_LOSS_WEIGHT="${TONE_LOSS_WEIGHT:-0.05}"
export TONE_TEMPERATURE="${TONE_TEMPERATURE:-0.1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python: $PYTHON_BIN. Set PYTHON_BIN to your Qwen3-ASR environment." >&2
  exit 1
fi
"$PYTHON_BIN" - <<'PY_CHECK'
import torch, torchaudio, pytorch_lightning
from transformers import Qwen3ASREncoder, AutoFeatureExtractor
if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
    raise SystemExit("Four visible CUDA GPUs are required; check CUDA_VISIBLE_DEVICES and the NVIDIA driver.")
print(f"Training environment: torch={torch.__version__}, CUDA={torch.version.cuda}, GPUs={torch.cuda.device_count()}")
PY_CHECK

if [[ "${FORCE_SPLIT:-0}" == "1" || ! -s "$SPLIT_DIR/train.jsonl" || ! -s "$SPLIT_DIR/dev.jsonl" ]]; then
  "$PYTHON_BIN" scripts/prepare_jsonl_ctc_splits.py \
    --input train.jsonl --data-root "$DATA_ROOT" --output-dir "$SPLIT_DIR" \
    --dev-ratio 0.03 --min-duration 0.3 --max-duration 20.0 --seed 42
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
"$PYTHON_BIN" scripts/baseline/finetuning_pinyin_ctc_qwen3.py \
  --data-root "$DATA_ROOT" \
  --train-path "$SPLIT_DIR/train.jsonl" \
  --val-path "$SPLIT_DIR/dev.jsonl" \
  --token-table-path data/lang_jsonl/tokens.txt \
  --model-name Qwen/Qwen3-ASR-0.6B-hf \
  --ctc-vocab 188 \
  --exp-dir "$EXP_DIR" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
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
  --tone-contrastive \
  --tone-acoustic-only \
  --final-bucket-sampling \
  --tone-start-step "$TONE_START_STEP" \
  --tone-ramp-steps "$TONE_RAMP_STEPS" \
  --tone-loss-weight "$TONE_LOSS_WEIGHT" \
  --tone-temperature "$TONE_TEMPERATURE" \
  --seed 42 \
  --devices 4 \
  --strategy ddp \
  "$@"

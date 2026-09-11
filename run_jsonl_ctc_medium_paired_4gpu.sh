#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/xintong/miniconda3/envs/whisper-pinyin/bin/python"
TORCH_LIB="$($PYTHON_BIN -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

DATA_ROOT="/data2/xintong/datasets/datasets"
CANONICAL_ROOT="/data2/xintong/datasets/cosyvoice3/cosyvoice3"
PAIRED_MANIFEST="data/mdd_paired_manifest/train.jsonl"
SPLIT_DIR="data/jsonl_actual_paired_split"
EXP_DIR="/data2/xintong/checkpoints/whisper-pinyin-ctc"

"$PYTHON_BIN" scripts/prepare_jsonl_ctc_splits.py \
  --input "$PAIRED_MANIFEST" \
  --data-root "$DATA_ROOT" \
  --output-dir "$SPLIT_DIR" \
  --dev-ratio 0.03 \
  --min-duration 0.3 \
  --max-duration 20.0 \
  --seed 42

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
"$PYTHON_BIN" scripts/baseline/finetuning_pinyin_ctc_torch.py \
  --data-root "$DATA_ROOT" \
  --canonical-root "$CANONICAL_ROOT" \
  --canonical-ctc-weight 1.0 \
  --train-name whisper_pinyin_ctc_medium_paired \
  --train-id 001 \
  --exp-dir "$EXP_DIR" \
  --train-path "$SPLIT_DIR/train.jsonl" \
  --val-path "$SPLIT_DIR/dev.jsonl" \
  --lexicon-path data/lang_jsonl/lexicon.txt \
  --token-table-path data/lang_jsonl/tokens.txt \
  --ctc-vocab 188 \
  --model-name medium \
  --ctc-layers 2 \
  --n_mels 80 \
  --batch-size 4 \
  --gradient-accumulation-steps 3 \
  --precision 16-mixed \
  --learning-rate 5e-5 \
  --weight-decay 0 \
  --adam-epsilon 1e-8 \
  --max-steps 30000 \
  --freeze-backbone-steps 10000 \
  --head-warmup-steps 3000 \
  --backbone-ramp-steps 3000 \
  --eval-steps 1000 \
  --dropout 0.1 \
  --seed 42 \
  --devices 4 \
  --strategy ddp

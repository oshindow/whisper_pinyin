#!/usr/bin/env bash
# Two-stage recipe with the original tonal-final units (188 tokens) plus an F0
# branch: WORLD harvest tracks are pre-extracted, encoded by a small conv stack
# and fused with the Whisper encoder output before the CTC head.
set -euo pipefail

PYTHON_BIN="/home/xintong/miniconda3/envs/whisper-pinyin/bin/python"
TORCH_LIB="$($PYTHON_BIN -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

DATA_ROOT="${DATA_ROOT:-/data2/xintong/mandarin_accent}"
SPLIT_DIR="data/jsonl_actual_split"
LANG_DIR="data/lang_jsonl"
F0_CACHE="${F0_CACHE:-/data2/xintong/f0_cache_world}"
F0_JOBS="${F0_JOBS:-32}"
EXP_DIR="/data2/xintong/checkpoints"

# The duration filter and the dev split do not depend on the recipe, so every
# run trains on exactly the same rows.
if [ -f "$SPLIT_DIR/train.jsonl" ] && [ -f "$SPLIT_DIR/dev.jsonl" ]; then
  echo "Reusing existing split in $SPLIT_DIR"
else
  "$PYTHON_BIN" scripts/prepare_jsonl_ctc_splits.py \
    --input train.jsonl \
    --data-root "$DATA_ROOT" \
    --output-dir "$SPLIT_DIR" \
    --dev-ratio 0.03 \
    --min-duration 0.3 \
    --max-duration 20.0 \
    --seed 42
fi

# Harvest runs at about four times real time per core; already cached
# utterances are skipped, so re-running this is cheap.
"$PYTHON_BIN" scripts/extract_f0.py \
  --manifests "$SPLIT_DIR/train.jsonl" "$SPLIT_DIR/dev.jsonl" test_seen.jsonl test_unseen.jsonl \
  --data-root "$DATA_ROOT" \
  --cache-dir "$F0_CACHE" \
  --jobs "$F0_JOBS"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
"$PYTHON_BIN" scripts/baseline/finetuning_pinyin_ctc_torch.py \
  --data-root "$DATA_ROOT" \
  --train-name whisper_pinyin_ctc_medium_actual_f0_twostage \
  --train-id 001 \
  --exp-dir "$EXP_DIR" \
  --train-path "$SPLIT_DIR/train.jsonl" \
  --val-path "$SPLIT_DIR/dev.jsonl" \
  --lexicon-path "$LANG_DIR/lexicon.txt" \
  --token-table-path "$LANG_DIR/tokens.txt" \
  --ctc-vocab 188 \
  --f0-cache-dir "$F0_CACHE" \
  --f0-dim 256 \
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

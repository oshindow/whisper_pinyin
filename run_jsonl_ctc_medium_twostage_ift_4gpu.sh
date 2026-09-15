#!/usr/bin/env bash
# Two-stage recipe with initial / final / tone CTC units: every tonal final is
# split into a toneless final plus a tone, so one syllable spans three units.
# The manifests are untouched; only the lexicon and token table differ from
# run_jsonl_ctc_medium_twostage_4gpu.sh.
set -euo pipefail

PYTHON_BIN="/home/xintong/miniconda3/envs/whisper-pinyin/bin/python"
TORCH_LIB="$($PYTHON_BIN -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

DATA_ROOT="${DATA_ROOT:-/data2/xintong/mandarin_accent}"
SPLIT_DIR="data/jsonl_actual_split"
LANG_DIR="data/lang_jsonl_ift"
EXP_DIR="/data2/xintong/checkpoints"

"$PYTHON_BIN" scripts/build_jsonl_lexicon_and_audit.py \
  --train-jsonl train.jsonl \
  --phone-field actual_phones \
  --unit ift \
  --data-root "$DATA_ROOT" \
  --lexicon-dir "$LANG_DIR" \
  --report-dir reports/audio_audit_ift

CTC_VOCAB="$(wc -l < "$LANG_DIR/tokens.txt")"
echo "CTC vocabulary: $CTC_VOCAB units"

# The duration filter and the dev split do not depend on the CTC units, so both
# recipes train on exactly the same rows.
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

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
"$PYTHON_BIN" scripts/baseline/finetuning_pinyin_ctc_torch.py \
  --data-root "$DATA_ROOT" \
  --train-name whisper_pinyin_ctc_medium_actual_ift_twostage \
  --train-id 001 \
  --exp-dir "$EXP_DIR" \
  --train-path "$SPLIT_DIR/train.jsonl" \
  --val-path "$SPLIT_DIR/dev.jsonl" \
  --lexicon-path "$LANG_DIR/lexicon.txt" \
  --token-table-path "$LANG_DIR/tokens.txt" \
  --ctc-vocab "$CTC_VOCAB" \
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

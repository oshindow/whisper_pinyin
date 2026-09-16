#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

python scripts/baseline/finetuning_pinyin_ctc_wav2vec2.py \
  --train-path data/jsonl_actual_split/train.jsonl \
  --val-path data/jsonl_actual_split/dev.jsonl \
  --data-root /data2/xintong/mandarin_accent \
  --token-table-path data/lang_jsonl/tokens.txt \
  --model-name TencentGameMate/chinese-wav2vec2-large \
  --ctc-vocab 188 \
  --exp-dir /data2/xintong/checkpoints/wav2vec2_large_actual_finalbucket_f0 \
  --batch-size 4 \
  --gradient-accumulation-steps 3 \
  --devices 4 \
  --strategy ddp_find_unused_parameters_true \
  --precision 16-mixed \
  --workers 8 \
  --learning-rate 5e-5 \
  --max-steps 30000 \
  --freeze-backbone-steps 10000 \
  --backbone-ramp-steps 3000 \
  --head-warmup-steps 3000 \
  --eval-steps 1000 \
  --mask-time-prob 0.65 \
  --mask-time-length 10 \
  --mask-feature-prob 0.25 \
  --mask-feature-length 64 \
  --f0-cache-dir /data2/xintong/f0_cache_world \
  --f0-dim 256 \
  --f0-loss-weight 0.05 \
  --final-bucket-sampling \
  --tone-contrastive \
  --tone-acoustic-only \
  --tone-loss-weight 0.05 \
  --tone-temperature 0.1 \
  --tone-start-step 10000 \
  --tone-ramp-steps 2000 \
  --tone-cross-accent-weight 2.0 \
  --seed 42 \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXP_DIR="${EXP_DIR:-/data2/xintong/checkpoints/qwen3_asr_ctc_actual_tonecontrast_finalbucket_acoustic}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-$EXP_DIR/checkpoints/last.ckpt}"

if [[ ! -s "$RESUME_CHECKPOINT" ]]; then
  echo "Checkpoint missing or empty: $RESUME_CHECKPOINT" >&2
  exit 1
fi

echo "Resuming training from: $RESUME_CHECKPOINT"
exec bash "$SCRIPT_DIR/run_jsonl_qwen3_ctc_finalbucket_4gpu.sh" \
  --resume-from-checkpoint "$RESUME_CHECKPOINT" "$@"

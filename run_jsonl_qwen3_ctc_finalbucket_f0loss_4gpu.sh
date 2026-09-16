#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
F0_CACHE="${F0_CACHE:-/data2/xintong/f0_cache_world}"
EXP_DIR="${EXP_DIR:-/data2/xintong/checkpoints/qwen3_asr_ctc_actual_finalbucket_f0input_f0loss}"

if [[ ! -d "$F0_CACHE" ]]; then
  echo "F0 cache missing: $F0_CACHE" >&2
  exit 1
fi

export EXP_DIR

exec bash "$SCRIPT_DIR/run_jsonl_qwen3_ctc_finalbucket_4gpu.sh" \
  --f0-cache-dir "$F0_CACHE" \
  --f0-dim 256 \
  --f0-loss-weight 0.05 \
  --mfa-alignment-dir "" \
  "$@"

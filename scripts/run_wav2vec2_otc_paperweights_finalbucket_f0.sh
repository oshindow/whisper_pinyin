#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export CONFIG_PATH="${CONFIG_PATH:-configs/wav2vec2_otc_paperweights_finalbucket_f0.yaml}"

cd "$REPO_ROOT"
exec bash scripts/run_wav2vec2_otc_finalbucket_f0.sh "$@"

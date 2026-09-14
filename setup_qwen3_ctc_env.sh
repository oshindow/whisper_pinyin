#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
"${BASE_PYTHON:-python3}" -m venv .venv-qwen3-ctc
PYTHON_BIN="$SCRIPT_DIR/.venv-qwen3-ctc/bin/python"
"$PYTHON_BIN" -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
"$PYTHON_BIN" -m pip install -r requirements-qwen3-ctc.txt
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" -c 'from transformers import Qwen3ASREncoder, AutoFeatureExtractor; import pytorch_lightning, torchaudio; import torch; print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "GPUs:", torch.cuda.device_count())'

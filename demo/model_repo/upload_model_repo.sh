#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <hf-namespace/model-name> <path-to-cross-continuous-ckpt>"
  echo "Example: $0 <hf-namespace>/whisper-pinyin /path/to/checkpoint-epoch=0009.ckpt"
  exit 1
fi

MODEL_ID="$1"
CKPT_PATH="$2"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Checkpoint not found: $CKPT_PATH"
  exit 1
fi

huggingface-cli upload "$MODEL_ID" demo/model_repo/README.md README.md --repo-type model
huggingface-cli upload "$MODEL_ID" demo/model_repo/config.json config.json --repo-type model
huggingface-cli upload "$MODEL_ID" demo/model_repo/configuration_whisper_pinyin.py configuration_whisper_pinyin.py --repo-type model
huggingface-cli upload "$MODEL_ID" demo/model_repo/modeling_whisper_pinyin.py modeling_whisper_pinyin.py --repo-type model
huggingface-cli upload "$MODEL_ID" demo/model_repo/.gitattributes .gitattributes --repo-type model
huggingface-cli upload "$MODEL_ID" data/whisper/tokens.txt tokens.txt --repo-type model
huggingface-cli upload "$MODEL_ID" "$CKPT_PATH" checkpoint-epoch=0009.ckpt --repo-type model

echo "Uploaded model repo to https://huggingface.co/$MODEL_ID"

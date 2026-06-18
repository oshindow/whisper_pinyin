#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <hf-namespace/space-name>"
  echo "Example: $0 walston/whisper-pinyin-demo"
  exit 1
fi

SPACE_ID="$1"
DEFAULT_AUDIO="demo/space/examples/example.wav"

python -m wave "$DEFAULT_AUDIO" >/dev/null

huggingface-cli upload "$SPACE_ID" demo/space/README.md README.md --repo-type space
huggingface-cli upload "$SPACE_ID" demo/space/app.py app.py --repo-type space
huggingface-cli upload "$SPACE_ID" demo/space/requirements.txt requirements.txt --repo-type space
huggingface-cli upload "$SPACE_ID" demo/space/packages.txt packages.txt --repo-type space
huggingface-cli upload "$SPACE_ID" demo/space/.gitattributes .gitattributes --repo-type space
huggingface-cli upload "$SPACE_ID" "$DEFAULT_AUDIO" examples/example.wav --repo-type space
huggingface-cli upload "$SPACE_ID" demo/space/examples examples --repo-type space
huggingface-cli upload "$SPACE_ID" whisper whisper --repo-type space
echo "Uploaded demo to https://huggingface.co/spaces/$SPACE_ID"

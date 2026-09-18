#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
F0_PYTHON_BIN="${F0_PYTHON_BIN:-/home/xintong/miniconda3/bin/python}"
TRAIN_PYTHON_BIN="${TRAIN_PYTHON_BIN:-/home/xintong/miniconda3/envs/whisper-pinyin/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-configs/wav2vec2_otc_finalbucket_f0.yaml}"

mapfile -t CONFIG_VALUES < <("$F0_PYTHON_BIN" - "$CONFIG_PATH" <<'PY_CONFIG'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
data = config["data"]
f0 = data["f0"]
split = data["split"]
print(data["data_root"])
print(f0["cache_dir"])
print(f0["extraction_jobs"])
print(config["training"]["devices"])
print(1 if split["enabled"] else 0)
print(split["input_manifest"])
print(split["output_dir"])
print(split["dev_ratio"])
print(split["seed"])
print(data["train_filter_seconds"][0])
print(data["train_filter_seconds"][1])
for manifest in (data["train_path"], data["val_path"], *data["test_manifests"]):
    print(manifest)
PY_CONFIG
)
DATA_ROOT="${CONFIG_VALUES[0]}"
F0_CACHE="${CONFIG_VALUES[1]}"
F0_JOBS="${CONFIG_VALUES[2]}"
REQUIRED_GPUS="${CONFIG_VALUES[3]}"
SPLIT_ENABLED="${CONFIG_VALUES[4]}"
SPLIT_INPUT="${CONFIG_VALUES[5]}"
SPLIT_OUTPUT_DIR="${CONFIG_VALUES[6]}"
SPLIT_DEV_RATIO="${CONFIG_VALUES[7]}"
SPLIT_SEED="${CONFIG_VALUES[8]}"
SPLIT_MIN_DURATION="${CONFIG_VALUES[9]}"
SPLIT_MAX_DURATION="${CONFIG_VALUES[10]}"
F0_MANIFESTS=("${CONFIG_VALUES[@]:11}")

# Extraction and OTC training currently use separate environments on this
# machine: the first provides pyworld, while the second provides k2.
"$F0_PYTHON_BIN" -c 'import numpy, pyworld, torch, torchaudio'
REQUIRED_GPUS="$REQUIRED_GPUS" "$TRAIN_PYTHON_BIN" - <<'PY_CHECK'
import os
import k2
import torch
import torchaudio

required = int(os.environ["REQUIRED_GPUS"])
available = torch.cuda.device_count() if torch.cuda.is_available() else 0
if available < required:
    raise SystemExit(
        f"Wav2Vec2 OTC training requires {required} CUDA GPUs, but PyTorch sees "
        f"{available}. Check nvidia-smi, the NVIDIA driver, and CUDA_VISIBLE_DEVICES."
    )
print(f"Training environment OK: torch={torch.__version__}, GPUs={available}")
PY_CHECK

if [[ "$SPLIT_ENABLED" == "1" ]]; then
  echo "Regenerating source-stratified train/dev split in $SPLIT_OUTPUT_DIR"
  "$F0_PYTHON_BIN" scripts/prepare_jsonl_ctc_splits.py \
    --input "$SPLIT_INPUT" \
    --data-root "$DATA_ROOT" \
    --output-dir "$SPLIT_OUTPUT_DIR" \
    --dev-ratio "$SPLIT_DEV_RATIO" \
    --min-duration "$SPLIT_MIN_DURATION" \
    --max-duration "$SPLIT_MAX_DURATION" \
    --seed "$SPLIT_SEED"
else
  echo "Reusing train/dev split from YAML: ${F0_MANIFESTS[0]}, ${F0_MANIFESTS[1]}"
fi

# Reuse the WORLD extractor. Existing cache files are skipped.
"$F0_PYTHON_BIN" scripts/extract_f0.py \
  --manifests "${F0_MANIFESTS[@]}" \
  --data-root "$DATA_ROOT" \
  --cache-dir "$F0_CACHE" \
  --jobs "$F0_JOBS"

"$TRAIN_PYTHON_BIN" scripts/baseline/finetuning_pinyin_otc_wav2vec2.py \
  --config "$CONFIG_PATH" \
  "$@"

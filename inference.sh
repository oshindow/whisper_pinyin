export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
CHECKPOINT="${CHECKPOINT:-/path/to/checkpoint.ckpt}"

CUDA_VISIBLE_DEVICES=0 python inference/inference_pinyin_ctc.py \
  --checkpoint "$CHECKPOINT" \
  --test-path dump/aishell3/test/text \
  --output results_aishell3_test.txt

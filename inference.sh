export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0 python inference/inference_pinyin_ctc.py \
  --checkpoint /data1/xintong/whisper_otc/exp2/whisper_pinyin_cross_ratio_random/001/checkpoint-epoch=0009.ckpt \
  --test-path dump/aishell3/test/text \
  --output results_aishell3_test.txt

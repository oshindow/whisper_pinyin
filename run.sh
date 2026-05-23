export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# CUDA_VISIBLE_DEVICES=0 python3 -u scripts/whisper_pinyin/finetuning_pinyin_otc_cross_fsq_mellen.py \
#     --data-root /data2/xintong \
#     --epoch 10 --train-name "whisper_pinyin" --train-id "001" \
#     --initial-bypass-weight -50 --initial-self-loop-weight 3.75 \
#     --batch-size 8 --train-path dump2/aishell3/train/text \
#     --model-name "small" --ctc-layers 2 --n_mels 80 --precision "bf16-mixed" \
#     --learning-rate 1e-4 --weight-decay 0.01 --adam-epsilon 1e-8 --warmup-steps 1000 \
#     > exp.log


CUDA_VISIBLE_DEVICES=0 python scripts/baseline/finetuning_pinyin_otc.py \
  --epoch 10 \
  --data-root /data2/xintong \
  --train-name whisper_pinyin_aishell3_otc \
  --train-id 001 \
  --train-path dump/aishell3/train/text \
  --model-name small \
  --ctc-layers 2 \
  --n_mels 80 \
  --batch-size 6 \
  --precision bf16-mixed \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --adam-epsilon 1e-8 \
  --warmup-steps 1000 > exp.log


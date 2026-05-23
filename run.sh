CUDA_VISIBLE_DEVICES=0 python3 -u finetuning_pinyin_otc_only_w2v.py \
    --epoch 10 --train-name "whisper_pinyin" --train-id "001" \
    --initial-bypass-weight -50 --initial-self-loop-weight 3.75 \
    --batch-size 8 --train-path resources/whisAID/multi_domain/train_all_w_o_accent \
    --model-name "small" --ctc-layers 2 --n_mels 80 --precision "bf16-mixed" \
    --learning-rate 1e-4 --weight-decay 0.01 --adam-epsilon 1e-8 --warmup-steps 1000 \
    > exp.log


 
# Whisper-Pinyin CTC: two-stage fine-tuning

The two-stage recipe trains Whisper Medium with actual-phone CTC targets:

- permanently frozen convolutional frontend;
- CTC-head-only phase for the first 10,000 optimizer steps;
- 3,000-step Transformer ramp followed by linear decay;
- 30,000 total optimizer steps, FP16, four-GPU DDP;
- source-stratified 3% development split and corpus-level PER selection.

Local JSONL manifests, audio, checkpoints, logs, generated splits, predictions,
and evaluation reports are intentionally excluded from Git.

Prepare the manifests and start training with:

```bash
./run_jsonl_ctc_medium_twostage_4gpu.sh
```

The launcher expects `train.jsonl` in the repository root and audio below
`/data2/xintong/datasets/datasets`. Checkpoints are written below
`/data2/xintong/checkpoints/whisper_pinyin_ctc_medium_actual_twostage/`.

Evaluate a checkpoint with `scripts/evaluate_jsonl_mdd.py`; pass the local
`test_seen.jsonl` and `test_unseen.jsonl` manifests via `--manifests`.

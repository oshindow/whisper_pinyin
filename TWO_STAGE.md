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

## F0 input with auxiliary F0 regression

Based on remote `tone` commit `1c25430`. This recipe retains the original
unsplit tonal-final CTC vocabulary and the remote F0 encoder/fusion branch.

```bash
./run_jsonl_ctc_medium_twostage_f0loss_4gpu.sh
# Override the auxiliary weight if needed:
F0_LOSS_WEIGHT=0.05 ./run_jsonl_ctc_medium_twostage_f0loss_4gpu.sh
```

The objective is `CTC + 0.1 * F0_SmoothL1` by default. A two-layer MLP predicts
utterance-normalized log-F0 from Whisper features **before F0 fusion**; this
prevents directly copying the supplied F0 input. Targets are channel 0 of the
existing cached WORLD features. Only voiced frames (channel 2) contribute;
unvoiced/interpolated and padded frames are excluded. An all-unvoiced batch
has differentiable zero auxiliary loss. No tone classification head is used.
This is contour regression, not absolute F0 in Hz; it does not supervise
voicing or delta-F0. The existing per-utterance normalization is unchanged.

The CLI flag `--f0-loss-weight` defaults to 0 (disabled), and positive weights
require `--f0-cache-dir`. The auxiliary head uses the head warmup/decay schedule.
The encoder follows the existing 10k freeze / 3k ramp schedule. Loss gradients
reach the Whisper encoder after unfreezing; the F0 input branch is trained by
CTC. Validation logs `val/f0_loss`, `val/f0_mae` (normalized log-F0 units), and
`val/f0_voiced_frames`, aggregated across ranks. Checkpoint selection remains
`val/per`. Experiment name: `whisper_pinyin_ctc_medium_actual_f0loss_twostage`.

Evaluate with `scripts/evaluate_jsonl_mdd.py`, using the original token table
and `--f0-cache-dir /data2/xintong/f0_cache_world --f0-dim 256`. The auxiliary
head is restored automatically from its weights but is not needed for CTC
decoding. F0 input remains required at inference, as in the remote recipe.

Data root defaults to `/data2/xintong/datasets/datasets`; the shared audio
resolver also maps legacy `LATIC/WAVE` paths to `/data2/xintong/LATIC/WAVA`.

F0 extraction dependencies (Python 3.8 environment):

```bash
/home/xintong/miniconda3/envs/whisper-pinyin/bin/python -m pip install Cython==0.29.37
/home/xintong/miniconda3/envs/whisper-pinyin/bin/python -m pip install --no-build-isolation -r requirements-f0.txt
```

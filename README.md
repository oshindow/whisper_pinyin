# 🎧 Whisper-Pinyin

[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0.0%2Bcu118-ee4c2c.svg)](https://pytorch.org/)
[![k2](https://img.shields.io/badge/k2-1.24.4.dev20250304-green.svg)](https://github.com/k2-fsa/k2)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Official implementation of **Whisper-Pinyin**, a Mandarin pinyin-level speech model for ASR and mispronunciation diagnosis.

This release is organized as an AISHELL-3-only open-source codebase:

- use the AISHELL-3 training set for fine-tuning;
- use the AISHELL-3 testing set for evaluation;

## 🚀 Quick Start

```bash
git clone https://github.com/oshindow/whisper_pinyin.git
cd whisper_pinyin

conda env create -f environment.yml
pip install pytorch-lightning==2.4.0 --no-deps
conda activate whisper-pinyin

python -c "import torch, torchaudio, k2; print(torch.__version__)"
```
 

## 🗂️ Repository Layout

```text
config.py                                      Shared experiment defaults
requirements.txt                              Python package versions
environment.yml                               Conda environment
utils.py                                      Metrics and helper functions
graph_compiler.py                             CTC graph compiler
otc_graph_compiler.py                         OTC graph compiler

preprocessing/
  preprocess.py                               Character-level dataset
  preprocess_pinyin.py                        Whisper-Pinyin dataset
  preprocess_pinyin_w2v.py                    Wav2Vec2 dataset

scripts/baseline/
  finetuning_pinyin_otc.py                    Baseline: OTC
  finetuning_pinyin_w2v.py                    Baseline: Wav2Vec2
  finetuning_pinyin_ctc_k2.py                 Baseline: CTC with k2
  finetuning_pinyin_ctc_torch.py              Baseline: CTC with torch

scripts/whisper_pinyin/
  finetuning_pinyin_otc_cross_continuous.py
  finetuning_pinyin_otc_cross_continuous_mellen.py
  finetuning_pinyin_otc_cross_fsq_mellen.py

scripts/annotator/
  finetuning.py                               Character annotator
  finetuning_pinyin.py                        Pinyin annotator

inference/
  inference_pinyin_ctc.py                     Whisper-Pinyin inference
  inference_pinyin_ctc_w2v.py                 Wav2Vec2 inference
  inference.py                                Character-level inference
  inference_pinyin.py                         Pinyin decoder inference
```
 

## 📦 Data Preparation

This repository already provides AISHELL-3 text manifests under:

```text
dump/aishell3/train/text
dump/aishell3/val/text
dump/aishell3/val/text_100
dump/aishell3/test/text
```

You only need to prepare the AISHELL-3 audio files. Convert all audio to 16 kHz, 16-bit, mono-channel WAV files and place them with the expected directory layout:

```text
<data-root>/aishell3/
  train/wav_16k/<speaker_id>/<utt_id>.wav
  test/wav_16k/<speaker_id>/<utt_id>.wav
```

`dump/aishell3/val/text_100` is a 100-utterance validation subset randomly sampled from `dump/aishell3/val/text`.

## 🏋️ Fine-Tuning

Baseline Whisper-OTC:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/baseline/finetuning_pinyin_otc.py \
  --epoch 10 \
  --data-root path/to/aishell3 \
  --train-name whisper_pinyin_aishell3_otc \
  --train-id 001 \
  --exp-dir exp2 \
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
```

Baseline Wav2Vec2:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/baseline/finetuning_pinyin_w2v.py \
  --epoch 10 \
  --data-root path/to/aishell3 \
  --train-name whisper_pinyin_aishell3_w2v \
  --train-id 001 \
  --exp-dir exp2 \
  --train-path dump/aishell3/train/text \
  --model-name small \
  --ctc-layers 2 \
  --n_mels 80 \
  --batch-size 6 \
  --precision bf16-mixed \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --adam-epsilon 1e-8 \
  --warmup-steps 1000 > exp_w2v.log
```

Whisper-Pinyin variants:

Whisper-Pinyin OTC cross continuous:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/whisper_pinyin/finetuning_pinyin_otc_cross_continuous.py \
  --epoch 10 \
  --data-root path/to/aishell3 \
  --train-name whisper_pinyin_aishell3_otc_cross_continuous \
  --train-id 001 \
  --exp-dir exp2 \
  --train-path dump/aishell3/train/text \
  --model-name small \
  --ctc-layers 2 \
  --n_mels 80 \
  --batch-size 6 \
  --precision bf16-mixed \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --adam-epsilon 1e-8 \
  --warmup-steps 1000 > exp_otc_cross_continuous.log
```

Whisper-Pinyin OTC cross continuous MELLEN:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/whisper_pinyin/finetuning_pinyin_otc_cross_continuous_mellen.py \
  --epoch 10 \
  --data-root path/to/aishell3 \
  --train-name whisper_pinyin_aishell3_otc_cross_continuous_mellen \
  --train-id 001 \
  --exp-dir exp2 \
  --train-path dump/aishell3/train/text \
  --model-name small \
  --ctc-layers 2 \
  --n_mels 80 \
  --batch-size 6 \
  --precision bf16-mixed \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --adam-epsilon 1e-8 \
  --warmup-steps 1000 > exp_otc_cross_continuous_mellen.log
```

Whisper-Pinyin OTC cross FSQ MELLEN:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/whisper_pinyin/finetuning_pinyin_otc_cross_fsq_mellen.py \
  --epoch 10 \
  --data-root path/to/aishell3 \
  --train-name whisper_pinyin_aishell3_otc_cross_fsq_mellen \
  --train-id 001 \
  --exp-dir exp2 \
  --train-path dump/aishell3/train/text \
  --model-name small \
  --ctc-layers 2 \
  --n_mels 80 \
  --batch-size 6 \
  --precision bf16-mixed \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --adam-epsilon 1e-8 \
  --warmup-steps 1000 > exp_otc_cross_fsq_mellen.log
```

Checkpoints are saved under:

```text
exp2/<train-name>/<train-id>/
```

## 🔍 Inference

Run Whisper-Pinyin inference on the AISHELL-3 test manifest:

```bash
CUDA_VISIBLE_DEVICES=0 python inference/inference_pinyin_ctc.py \
  --checkpoint path/to/your/checkpoint \
  --test-path dump/aishell3/test/text \
  --output results_aishell3_test.txt
```

Run Wav2Vec2 baseline inference:

```bash
CUDA_VISIBLE_DEVICES=0 python inference/inference_pinyin_ctc_w2v.py \
  --checkpoint path/to/your/checkpoint \
  --test-path dump/aishell3/test/text \
  --output results_w2v_aishell3_test.txt
```
  

## 📝 Notes for Users

- This project borrows and adapts a lot of code and ideas from [Whisper](https://github.com/openai/whisper), [k2](https://github.com/k2-fsa/k2), [icefall](https://github.com/k2-fsa/icefall), [SpeechBrain](https://github.com/speechbrain/speechbrain), and [FSQ](https://arxiv.org/abs/2309.15505). Please also follow the licenses and citation guidance of those upstream projects when using this repository.
- See [LICENSE](LICENSE) for the project license, [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream code notices, and `LICENSES/` for third-party license references.

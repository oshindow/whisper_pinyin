This is the official implementation of Whisper-Pinyin.
 
# Pre-trained models

The released checkpoints can be hosted on the Hugging Face Hub as model repositories. A recommended repository layout is:

```text
whisper-pinyin/
+-- README.md
+-- config.py
+-- checkpoint-epoch=0009.ckpt
+-- data/
|   +-- whisper/tokens.txt
|   +-- lang_whisper/lexicon_add_plus_new.txt
+-- inference_pinyin_ctc.py
```

Suggested model variants:

| Model | Description | Checkpoint |
| --- | --- | --- |
| `whisper-pinyin-latic` | LATIC fine-tuned Whisper-Pinyin model | `exp2/whisper_pinyin_sys/001/checkpoint-epoch=0009.ckpt` |
| `whisper-pinyin-magichub-sg` | Magichub-SG fine-tuned Whisper-Pinyin model | `exp2/whisper_pinyin_sys/005/checkpoint-epoch=0009.ckpt` |
| `whisper-pinyin-multidomain` | Multi-domain Whisper-Pinyin model | `exp2/whisper_pinyin_sys/020/checkpoint-epoch=0004.ckpt` |

Replace the checkpoint paths above with the final checkpoints you want to publish.

# Environment

Create and activate a Python environment:

```bash
conda create -n whisper-pinyin python=3.10 -y
conda activate whisper-pinyin
```

Install the core dependencies:

```bash
pip install torch torchaudio pytorch-lightning transformers openai-whisper gradio huggingface_hub
```

Install `k2` and `icefall` according to your CUDA and PyTorch versions. After installation, expose the local `icefall` package and shared libraries:

```bash
export PYTHONPATH=/path/to/icefall:$PYTHONPATH
export LD_LIBRARY_PATH=/path/to/conda/envs/whisper-pinyin/lib:$LD_LIBRARY_PATH
```

The training and decoding scripts expect the following project resources:

```text
data/whisper/tokens.txt
data/lang_whisper/lexicon_add_plus_new.txt
dump2/*/train/text
dump2/*/test/text
```

# Inference

Set the checkpoint path in `inference_pinyin_ctc.py`:

```python
model_path = "exp2/whisper_pinyin_sys/020/checkpoint-epoch=0004.ckpt"
```

Run inference:

```bash
export PYTHONPATH=/path/to/icefall:$PYTHONPATH
export LD_LIBRARY_PATH=/path/to/conda/envs/whisper-pinyin/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 python3 inference_pinyin_ctc.py
```

For reproducible experiments, keep the tokenizer, pinyin token table, lexicon, and checkpoint from the same training run.

# Fine-tuning

Fine-tune an OTC-based Whisper-Pinyin model:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u finetuning_pinyin_otc_only.py \
  --epoch 10 \
  --train-name "whisper_pinyin_multidomain" \
  --train-id "001" \
  --train-path "dump2/train_latic_sg_sichuan_coarse_format" \
  --model-name "small" \
  --ctc-layers 2 \
  --n_mels 80 \
  --batch-size 6 \
  --precision "bf16-mixed" \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --adam-epsilon 1e-8 \
  --warmup-steps 20 \
  --initial-bypass-weight -50 \
  --initial-self-loop-weight 3.75
```

Checkpoints are written to:

```text
/data1/xintong/whisper_otc/exp2/<train-name>/<train-id>/
```

TensorBoard logs are written to:

```text
/data1/xintong/whisper_otc/exp2/<train-name>/logs/<train-id>/
```

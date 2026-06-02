---
title: Whisper-Pinyin Demo
emoji: 🎧
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.34.2
python_version: 3.10.0
app_file: app.py
pinned: false
license: mit
---

# Whisper-Pinyin Demo

This Space hosts a Gradio demo for the Whisper-Pinyin cross-augmentation continuous checkpoint.

The checkpoint is stored in the model repository:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "walston/whisper-pinyin",
    trust_remote_code=True,
)
```

The app expects:

- `app.py`
- `requirements.txt`
- `packages.txt`
- `whisper/`

Use `demo/space/upload_space.sh` from the repository root to upload the Space app. Use `demo/model_repo/upload_model_repo.sh` to upload the checkpoint to the model repo.

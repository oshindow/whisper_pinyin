---
license: mit
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
  - whisper
  - pinyin
  - speech-recognition
  - mandarin
---

# Whisper-Pinyin

Cross-augmentation continuous Whisper-Pinyin checkpoint.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "walston/whisper-pinyin",
    trust_remote_code=True,
)
```

This repository stores the checkpoint separately from the demo Space. The demo Space loads this model repo at runtime.

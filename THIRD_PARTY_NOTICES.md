# Third-Party Notices

This project is released under the MIT License, but it includes or adapts code
and ideas from several upstream open-source projects. Please also follow the
licenses, citation guidance, and model/data terms of the upstream projects.

## OpenAI Whisper

- URL: https://github.com/openai/whisper
- License: MIT License
- Copyright: Copyright (c) 2022 OpenAI
- Usage in this repository: the local `whisper/` package is based on OpenAI
  Whisper and includes project-specific modifications for Whisper-Pinyin
  training and inference.
- License notice: see `LICENSES/OpenAI-Whisper-MIT.txt`.

## k2

- URL: https://github.com/k2-fsa/k2
- License: Apache License 2.0
- Usage in this repository: finite-state automata operations, graph
  compilation, and CTC/OTC loss computation.
- License notice: see `LICENSES/Apache-2.0.txt`.

## icefall

- URL: https://github.com/k2-fsa/icefall
- License: Apache License 2.0
- Usage in this repository: graph compiler code and training utilities are
  adapted from the k2/icefall ecosystem. The files `graph_compiler.py` and
  `otc_graph_compiler.py` retain Apache-2.0 license headers from their
  upstream sources.
- License notice: see `LICENSES/Apache-2.0.txt`.

## SpeechBrain

- URL: https://github.com/speechbrain/speechbrain
- License: Apache License 2.0
- Usage in this repository: this project borrows ideas and implementation
  patterns from SpeechBrain-style speech training components.
- License notice: see `LICENSES/Apache-2.0.txt`.

## FSQ / Finite Scalar Quantization

- Paper: https://arxiv.org/abs/2309.15505
- Related PyTorch implementation: https://github.com/lucidrains/vector-quantize-pytorch
- Related implementation license: MIT License
- Usage in this repository: `scripts/whisper_pinyin/fsq.py` implements finite
  scalar quantization for Whisper-Pinyin experiments and notes that it is
  adapted from the JAX version in the FSQ paper appendix.

## Notes

- Apache-2.0-licensed files keep their original license headers.
- MIT-licensed adapted files keep their original license notices where present.
- Dependency packages listed in `requirements.txt` are governed by their own
  upstream licenses.

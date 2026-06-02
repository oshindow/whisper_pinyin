from pathlib import Path
from typing import Dict, List

import torch
from huggingface_hub import hf_hub_download
from transformers import PreTrainedModel

from .configuration_whisper_pinyin import WhisperPinyinConfig


TONE_TOKENS = {"1", "2", "3", "4", "5"}


class WhisperPinyinModel(PreTrainedModel):
    config_class = WhisperPinyinConfig
    base_model_prefix = "whisper_pinyin"

    def __init__(self, config: WhisperPinyinConfig):
        super().__init__(config)
        self.whisper_model = None
        self.id_to_symbol = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        revision = kwargs.get("revision")
        cache_dir = kwargs.get("cache_dir")
        local_files_only = kwargs.get("local_files_only", False)
        token = kwargs.get("token", kwargs.get("use_auth_token"))

        if config is None:
            config = WhisperPinyinConfig.from_pretrained(
                pretrained_model_name_or_path,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                token=token,
            )

        model = cls(config)
        checkpoint_path = model._resolve_file(
            pretrained_model_name_or_path,
            config.checkpoint_filename,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
        token_table_path = model._resolve_file(
            pretrained_model_name_or_path,
            config.token_table_filename,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )

        import whisper

        model.whisper_model = whisper.load_model(str(checkpoint_path), device="cpu")
        model.whisper_model.eval()
        model.id_to_symbol = model._load_token_table(Path(token_table_path))
        return model

    def _resolve_file(
        self,
        repo_id_or_path,
        filename,
        revision=None,
        cache_dir=None,
        local_files_only=False,
        token=None,
    ):
        repo_path = Path(str(repo_id_or_path))
        if repo_path.is_dir():
            local_file = repo_path / filename
            if not local_file.is_file():
                raise FileNotFoundError(f"Could not find {filename} in {repo_path}")
            return local_file

        return hf_hub_download(
            repo_id=str(repo_id_or_path),
            filename=filename,
            repo_type="model",
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )

    def _load_token_table(self, path: Path) -> List[str]:
        id_to_symbol = {}
        with path.open("r", encoding="utf-8") as token_file:
            for line in token_file:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                symbol, index = parts
                id_to_symbol[int(index)] = symbol

        if not id_to_symbol:
            raise RuntimeError(f"No tokens found in {path}")

        return [id_to_symbol.get(i, "<unk>") for i in range(max(id_to_symbol) + 1)]

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.whisper_model is not None:
            self.whisper_model.to(*args, **kwargs)
        return self

    def eval(self):
        super().eval()
        if self.whisper_model is not None:
            self.whisper_model.eval()
        return self

    @torch.no_grad()
    def decode_file(self, audio_path: str, device: str = None) -> Dict[str, str]:
        if self.whisper_model is None or self.id_to_symbol is None:
            raise RuntimeError("Model is not loaded. Use AutoModel.from_pretrained first.")

        import whisper

        if device is None:
            device = next(self.whisper_model.parameters()).device

        audio = whisper.load_audio(audio_path, sr=self.config.sample_rate)
        duration = len(audio) / self.config.sample_rate
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.config.n_mels).to(device)

        encoded = self.whisper_model.encoder(mel.unsqueeze(0))
        audio_features = encoded[0] if isinstance(encoded, tuple) else encoded
        _, logits = self.whisper_model.ctc_head(audio_features)

        tokens = self._ctc_decode(logits)
        token_text = " ".join(tokens)
        return {
            "pinyin": self._format_syllables(tokens),
            "tokens": token_text,
            "duration": duration,
        }

    def _ctc_decode(self, logits: torch.Tensor) -> List[str]:
        token_ids = logits.argmax(dim=-1).squeeze(0).detach().cpu().tolist()
        non_blank_ids = [token_id for token_id in token_ids if token_id != 0]

        collapsed_ids = []
        previous_id = None
        for token_id in non_blank_ids:
            if token_id != previous_id:
                collapsed_ids.append(token_id)
                previous_id = token_id

        return [
            self.id_to_symbol[token_id] if token_id < len(self.id_to_symbol) else "<unk>"
            for token_id in collapsed_ids
            if token_id != 1
        ]

    def _format_syllables(self, tokens: List[str]) -> str:
        syllables = []
        current = []

        for token in tokens:
            current.append(token)
            if token in TONE_TOKENS:
                syllables.append("".join(current))
                current = []

        syllables.extend(current)
        return " ".join(syllables)

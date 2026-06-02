from transformers import PretrainedConfig


class WhisperPinyinConfig(PretrainedConfig):
    model_type = "whisper_pinyin"

    def __init__(
        self,
        checkpoint_filename="checkpoint-epoch=0009.ckpt",
        token_table_filename="tokens.txt",
        n_mels=80,
        sample_rate=16000,
        vocab_size=280,
        ctc_layers=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.checkpoint_filename = checkpoint_filename
        self.token_table_filename = token_table_filename
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.vocab_size = vocab_size
        self.ctc_layers = ctc_layers

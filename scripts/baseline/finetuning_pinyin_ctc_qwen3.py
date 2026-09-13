#!/usr/bin/env python3
"""Two-stage SpecAugment fine-tuning of Qwen3-ASR's audio encoder with CTC."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
import torch.nn as nn
import torchaudio
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset


def _qwen_imports():
    try:
        from transformers import AutoConfig, AutoFeatureExtractor, Qwen3ASREncoder
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3-ASR is unavailable in this transformers installation. "
            "Install a transformers release that contains Qwen3ASREncoder."
        ) from exc
    return AutoConfig, AutoFeatureExtractor, Qwen3ASREncoder


def load_audio_encoder(model_id: str):
    """Load only model.audio_tower.* tensors; never instantiate the text model."""
    AutoConfig, _, Qwen3ASREncoder = _qwen_imports()
    try:
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("huggingface_hub and safetensors are required") from exc

    config = AutoConfig.from_pretrained(model_id)
    encoder = Qwen3ASREncoder(config.audio_config)
    snapshot = Path(snapshot_download(
        model_id,
        allow_patterns=["config.json", "model*.safetensors"],
    ))
    tensor_files = sorted(snapshot.glob("model*.safetensors"))
    if not tensor_files:
        raise FileNotFoundError(f"No safetensors weights found in {snapshot}")

    destination = dict(encoder.named_parameters())
    destination.update(dict(encoder.named_buffers()))
    loaded = set()
    prefix = "model.audio_tower."
    with torch.no_grad():
        for filename in tensor_files:
            with safe_open(filename, framework="pt", device="cpu") as weights:
                for source_name in weights.keys():
                    if not source_name.startswith(prefix):
                        continue
                    name = source_name[len(prefix):]
                    if name in destination:
                        value = weights.get_tensor(source_name)
                        if destination[name].shape != value.shape:
                            raise ValueError(f"Shape mismatch for {name}: {value.shape} != {destination[name].shape}")
                        destination[name].copy_(value)
                        loaded.add(name)
    missing = set(dict(encoder.named_parameters())) - loaded
    if missing:
        raise RuntimeError(f"Missing {len(missing)} audio encoder tensors, e.g. {sorted(missing)[:5]}")
    print(f"Loaded {len(loaded)} Qwen3-ASR audio-encoder tensors only")
    return encoder


class PhoneTable:
    def __init__(self, path: str):
        self.id_to_phone = {}
        self.phone_to_id = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            phone, index = line.rsplit(maxsplit=1)
            index = int(index)
            self.id_to_phone[index] = phone
            self.phone_to_id[phone] = index

    def __len__(self):
        return len(self.phone_to_id)


class QwenCTCDataset(Dataset):
    def __init__(self, manifest: str, data_root: str, audio_field: str = "l2_wav"):
        self.audio_field = audio_field
        self.rows = []
        for line in Path(manifest).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if audio_field not in row:
                    raise KeyError(f"Manifest row is missing audio field {audio_field!r}")
                row[audio_field] = str(self._resolve_audio(row[audio_field], data_root))
                self.rows.append(row)

    @staticmethod
    def _resolve_audio(audio_path: str, data_root: str) -> Path:
        path = Path(audio_path)
        if path.is_file():
            return path
        marker = "/datasets/datasets/"
        normalized = str(path).replace("\\", "/")
        if marker in normalized:
            relative = normalized.split(marker, 1)[1]
            candidates = [Path(data_root) / relative]
            if relative == "LATIC" or relative.startswith("LATIC/"):
                candidates.append(Path(data_root) / "magichub_multiaccent" / relative)
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
        return path

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        waveform, sample_rate = torchaudio.load(row[self.audio_field])
        waveform = waveform.mean(0)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        return row, waveform.numpy()


class QwenCollator:
    def __init__(self, model_id: str):
        _, AutoFeatureExtractor, _ = _qwen_imports()
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.chunk_size = self.extractor.n_window * 2

    def __call__(self, examples):
        rows, waveforms = zip(*examples)
        encoded = self.extractor(
            list(waveforms), sampling_rate=16000, padding=True,
            return_attention_mask=True, return_tensors="pt",
        )
        features = encoded["input_features"]
        mask = encoded.get("input_features_mask", encoded.get("attention_mask"))
        if mask is None:
            raise KeyError("Qwen3 feature extractor did not return an input feature mask")

        # The encoder chunks mel frames in blocks of n_window * 2 (=100 by default).
        padded = math.ceil(features.shape[-1] / self.chunk_size) * self.chunk_size
        if padded != features.shape[-1]:
            features = nn.functional.pad(features, (0, padded - features.shape[-1]))
            mask = nn.functional.pad(mask, (0, padded - mask.shape[-1]))
        return list(rows), features, mask.long()


class PackedSpecAugment:
    """SpecAugment at the first Transformer layer input (after the CNN stack)."""
    def __init__(self, time_prob=.65, time_length=10, feature_prob=.25, feature_length=64):
        self.time_prob = time_prob
        self.time_length = time_length
        self.feature_prob = feature_prob
        self.feature_length = feature_length

    @staticmethod
    def _mask_spans(x, axis, size, probability):
        length = x.shape[axis]
        if length < size or probability <= 0:
            return
        count = max(1, int(probability * length / size + torch.rand((), device=x.device).item()))
        starts = torch.randint(0, length - size + 1, (count,), device=x.device)
        for start in starts.tolist():
            slices = [slice(None)] * x.ndim
            slices[axis] = slice(start, start + size)
            x[tuple(slices)] = 0

    def __call__(self, module, args, kwargs):
        if not module.training:
            return args, kwargs
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        cu_seqlens = kwargs.get("cu_seqlens", args[1] if len(args) > 1 else None)
        if hidden is None or cu_seqlens is None:
            return args, kwargs
        augmented = hidden.clone()
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            window = augmented[start:end]
            self._mask_spans(window, 0, self.time_length, self.time_prob)
            self._mask_spans(window, 1, self.feature_length, self.feature_prob)
        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = augmented
        else:
            args = (augmented, *args[1:])
        return args, kwargs


class Qwen3CTCModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.args = args
        self.phones = PhoneTable(args.token_table_path)
        if len(self.phones) != args.ctc_vocab:
            raise ValueError(f"Token table has {len(self.phones)} symbols, expected {args.ctc_vocab}")
        if self.phones.id_to_phone.get(0) not in {"<blk>", "<blank>", "<pad>"}:
            print(f"Warning: token 0 ({self.phones.id_to_phone.get(0)!r}) is used as CTC blank")

        self.encoder = load_audio_encoder(args.model_name)
        if self.encoder.config.num_mel_bins != 128:
            raise ValueError(f"This recipe expects 128 mel bins, got {self.encoder.config.num_mel_bins}")
        print(
            f"Audio encoder: {len(self.encoder.layers)} layers, "
            f"d_model={self.encoder.config.d_model}, "
            f"ffn_dim={self.encoder.config.encoder_ffn_dim}"
        )
        self.ctc_head = nn.Linear(self.encoder.config.d_model, args.ctc_vocab)
        nn.init.normal_(self.ctc_head.weight, mean=0.0, std=self.encoder.config.initializer_range)
        nn.init.zeros_(self.ctc_head.bias)
        for name in ("conv2d1", "conv2d2", "conv2d3", "conv_out"):
            for parameter in getattr(self.encoder, name).parameters():
                parameter.requires_grad = False
        augmenter = PackedSpecAugment(
            args.mask_time_prob, args.mask_time_length,
            args.mask_feature_prob, args.mask_feature_length,
        )
        self.encoder.layers[0].register_forward_pre_hook(augmenter, with_kwargs=True)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.register_buffer("val_errors", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("val_phones", torch.tensor(0, dtype=torch.long), persistent=False)

    def encode_hidden(self, features, feature_mask):
        lengths = feature_mask.sum(-1).long()
        packed = self.encoder(
            input_features=features, input_features_mask=feature_mask,
            return_dict=True,
        ).last_hidden_state
        output_lengths = self.encoder._post_cnn_length(lengths.remainder(self.encoder.n_window * 2))
        output_lengths += lengths.div(self.encoder.n_window * 2, rounding_mode="floor") * 13
        sequences = packed.split(output_lengths.tolist())
        hidden = pad_sequence(sequences, batch_first=True)
        return hidden, output_lengths

    def encode(self, features, feature_mask):
        hidden, output_lengths = self.encode_hidden(features, feature_mask)
        return self.ctc_head(hidden), output_lengths

    def targets(self, rows):
        labels = []
        for row in rows:
            phones = [p for p in row["actual_phones"] if p not in (None, "sil")]
            unknown = [p for p in phones if p not in self.phones.phone_to_id]
            if unknown:
                raise KeyError(f"Phones absent from token table: {sorted(set(unknown))}")
            labels.append([self.phones.phone_to_id[p] for p in phones])
        flat = torch.tensor([x for seq in labels for x in seq], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(seq) for seq in labels], dtype=torch.long, device=self.device)
        return flat, lengths, labels

    def training_step(self, batch, _):
        rows, features, mask = batch
        logits, output_lengths = self.encode(features, mask)
        targets, target_lengths, _ = self.targets(rows)
        loss = self.ctc_loss(
            logits.log_softmax(-1).transpose(0, 1), targets,
            output_lengths, target_lengths,
        )
        self.log("train/ctc_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    @staticmethod
    def edit_distance(reference, hypothesis):
        previous = list(range(len(hypothesis) + 1))
        for i, ref in enumerate(reference, 1):
            current = [i]
            for j, hyp in enumerate(hypothesis, 1):
                current.append(min(current[-1] + 1, previous[j] + 1,
                                   previous[j - 1] + (ref != hyp)))
            previous = current
        return previous[-1]

    def validation_step(self, batch, _):
        rows, features, mask = batch
        logits, output_lengths = self.encode(features, mask)
        _, _, references = self.targets(rows)
        predictions = logits.argmax(-1)
        for sequence, length, reference in zip(predictions, output_lengths, references):
            hypothesis, previous = [], -1
            for token in sequence[:int(length)]:
                token = int(token)
                if token != previous and token != 0:
                    hypothesis.append(token)
                previous = token
            self.val_errors += self.edit_distance(reference, hypothesis)
            self.val_phones += len(reference)

    def on_validation_epoch_start(self):
        self.val_errors.zero_()
        self.val_phones.zero_()

    def on_validation_epoch_end(self):
        totals = torch.stack((self.val_errors, self.val_phones))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        per = totals[0].float() / totals[1].clamp_min(1)
        self.log("val/per", per, prog_bar=True, logger=True, sync_dist=False)

    def configure_optimizers(self):
        backbone = [p for p in self.encoder.parameters() if p.requires_grad]
        optimizer = AdamW(
            [{"params": backbone, "name": "backbone"},
             {"params": self.ctc_head.parameters(), "name": "ctc_head"}],
            lr=self.args.learning_rate, betas=(0.9, 0.98),
            eps=self.args.adam_epsilon, weight_decay=0.0,
        )

        def head_lr(step):
            if step < self.args.head_warmup_steps:
                return step / max(1, self.args.head_warmup_steps)
            return max(0.0, (self.args.max_steps - step) /
                       max(1, self.args.max_steps - self.args.head_warmup_steps))

        def backbone_lr(step):
            if step < self.args.freeze_backbone_steps:
                return 0.0
            ramp_end = self.args.freeze_backbone_steps + self.args.backbone_ramp_steps
            if step < ramp_end:
                return (step - self.args.freeze_backbone_steps) / max(1, self.args.backbone_ramp_steps)
            return max(0.0, (self.args.max_steps - step) / max(1, self.args.max_steps - ramp_end))

        scheduler = LambdaLR(optimizer, lr_lambda=[backbone_lr, head_lr])
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--token-table-path", default="data/lang_jsonl/tokens.txt")
    parser.add_argument("--model-name", default="Qwen/Qwen3-ASR-0.6B-hf")
    parser.add_argument("--ctc-vocab", type=int, default=188)
    parser.add_argument("--exp-dir", default="exp/qwen3_asr_ctc")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=3)
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--strategy", default="ddp")
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--freeze-backbone-steps", type=int, default=10000)
    parser.add_argument("--backbone-ramp-steps", type=int, default=3000)
    parser.add_argument("--head-warmup-steps", type=int, default=3000)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--mask-time-prob", type=float, default=.65)
    parser.add_argument("--mask-time-length", type=int, default=10)
    parser.add_argument("--mask-feature-prob", type=float, default=.25)
    parser.add_argument("--mask-feature-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)
    Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    collator = QwenCollator(args.model_name)
    train_loader = DataLoader(
        QwenCTCDataset(args.train_path, args.data_root), batch_size=args.batch_size,
        shuffle=True, drop_last=True, num_workers=args.workers, collate_fn=collator,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        QwenCTCDataset(args.val_path, args.data_root), batch_size=args.batch_size,
        num_workers=args.workers, collate_fn=collator,
        persistent_workers=args.workers > 0,
    )
    checkpoint = ModelCheckpoint(
        dirpath=f"{args.exp_dir}/checkpoints",
        filename="checkpoint-step={step:06d}-per={val/per:.4f}", monitor="val/per",
        mode="min", save_top_k=2, save_last=True, auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices if torch.cuda.is_available() else 1,
        strategy=args.strategy if torch.cuda.is_available() else "auto",
        precision=args.precision, max_steps=args.max_steps, max_epochs=-1,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=1.0, gradient_clip_algorithm="norm",
        limit_train_batches=args.eval_steps * args.gradient_accumulation_steps,
        val_check_interval=1.0, check_val_every_n_epoch=1,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
        logger=TensorBoardLogger(args.exp_dir, name="logs"),
    )
    trainer.fit(Qwen3CTCModule(args), train_loader, val_loader)


if __name__ == "__main__":
    main()

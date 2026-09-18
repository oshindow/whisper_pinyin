#!/usr/bin/env python3
"""Wav2Vec2 k2 OTC training with final-bucket contrastive learning and F0."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import k2
import yaml
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoFeatureExtractor, Wav2Vec2Model

from scripts.baseline.final_bucket_sampler import FinalBucketSampler
from scripts.baseline.graceful_checkpoint import GracefulCheckpoint
from scripts.baseline.mfa_pooling import attach_mfa_intervals
from otc_graph_compiler import OtcTrainingGraphCompiler
from utils import str2bool
from scripts.baseline.finetuning_pinyin_ctc_qwen3 import (
    F0Encoder,
    F0Fusion,
    F0PredictionHead,
    PhoneTable,
    Qwen3CTCModule,
    QwenCTCDataset,
)


class Wav2Vec2Collator:
    """Pad raw 16 kHz waveforms; F0 remains per-utterance until encoder alignment."""

    def __init__(self, model_name: str, use_f0: bool = False):
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.use_f0 = use_f0

    def __call__(self, examples):
        if self.use_f0:
            rows, waveforms, f0s = zip(*examples)
        else:
            rows, waveforms = zip(*examples)
        encoded = self.extractor(
            list(waveforms), sampling_rate=16000, padding=True,
            return_attention_mask=True, return_tensors="pt",
        )
        batch = (list(rows), encoded["input_values"], encoded["attention_mask"].long())
        if self.use_f0:
            batch += ([torch.from_numpy(f0) for f0 in f0s],)
        return batch


class Wav2Vec2CTCModule(Qwen3CTCModule):
    """Reuse the established CTC/F0/final-contrastive losses with Wav2Vec2."""

    def __init__(self, args):
        # Deliberately do not call Qwen3CTCModule.__init__: it constructs Qwen3.
        LightningModule.__init__(self)
        self.save_hyperparameters(vars(args))
        self.args = args
        self.phones = PhoneTable(args.token_table_path)
        if len(self.phones) != args.ctc_vocab:
            raise ValueError(
                f"Token table has {len(self.phones)} symbols, expected {args.ctc_vocab}"
            )

        # Transformers applies these masks after convolution/feature projection and
        # before the Transformer encoder, only while model.training is true.
        self.encoder = Wav2Vec2Model.from_pretrained(
            args.model_name,
            apply_spec_augment=True,
            mask_time_prob=args.mask_time_prob,
            mask_time_length=args.mask_time_length,
            mask_feature_prob=args.mask_feature_prob,
            mask_feature_length=args.mask_feature_length,
        )
        hidden_size = self.encoder.config.hidden_size
        self.ctc_head = nn.Linear(hidden_size, args.ctc_vocab)
        nn.init.normal_(self.ctc_head.weight, 0.0, self.encoder.config.initializer_range)
        nn.init.zeros_(self.ctc_head.bias)

        # The raw-waveform convolutional frontend is kept fixed, matching common
        # Wav2Vec2 fine-tuning practice; Transformer layers follow the LR schedule.
        self.encoder.freeze_feature_encoder()

        self.use_f0 = args.f0_cache_dir is not None
        self.f0_loss_weight = args.f0_loss_weight
        if self.use_f0:
            self.f0_encoder = F0Encoder(out_dim=args.f0_dim)
            self.f0_fusion = F0Fusion(hidden_size, args.f0_dim)
            self.f0_head = F0PredictionHead(hidden_size)
        else:
            self.f0_encoder = self.f0_fusion = self.f0_head = None

        self.use_tone_contrastive = args.tone_contrastive
        self.tone_acoustic_only = args.tone_acoustic_only
        if self.use_tone_contrastive and not self.tone_acoustic_only:
            self.phone_embedding = nn.Embedding(args.ctc_vocab, hidden_size)
            nn.init.normal_(self.phone_embedding.weight, 0.0, self.encoder.config.initializer_range)

        self.loss_name = "otc"
        self.otc_graph_compiler = OtcTrainingGraphCompiler(
            Path(args.otc_token_table_path).parent,
            otc_token=args.otc_token,
            initial_bypass_weight=args.initial_bypass_weight,
            initial_self_loop_weight=args.initial_self_loop_weight,
            bypass_weight_decay=args.bypass_weight_decay,
            self_loop_weight_decay=args.self_loop_weight_decay,
        )
        self.register_buffer("val_errors", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("val_phones", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("val_f0_stats", torch.zeros(3, dtype=torch.float64), persistent=False)

        print(
            f"Wav2Vec2: layers={self.encoder.config.num_hidden_layers}, "
            f"hidden={hidden_size}; SpecAugment time={args.mask_time_prob}x{args.mask_time_length}, "
            f"feature={args.mask_feature_prob}x{args.mask_feature_length}",
            flush=True,
        )

    def encode_hidden(self, input_values, attention_mask):
        input_lengths = attention_mask.sum(-1).long()
        output_lengths = self.encoder._get_feat_extract_output_lengths(input_lengths).long()
        hidden = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        return hidden, output_lengths

    def ctc_loss(self, log_probs, targets, input_lengths, target_lengths):
        """Drop-in k2 OTC loss for the inherited training step."""
        # k2's DenseFsaVec/intersect_dense does not support the FP16 tensor
        # produced by Lightning's 16-mixed autocast. Keep the encoder/head in
        # mixed precision, but run the graph loss in FP32.
        batch_log_probs = log_probs.transpose(0, 1).float()
        nonblank = torch.logsumexp(batch_log_probs[..., 1:], dim=-1, keepdim=True)
        nonblank = nonblank - math.log(batch_log_probs.shape[-1] - 1)
        otc_log_probs = torch.cat((batch_log_probs, nonblank), dim=-1)

        lengths = target_lengths.tolist()
        sequences = [part.tolist() for part in targets.split(lengths)]

        # k2.intersect_dense requires DenseFsaVec sequences in non-increasing
        # frame-length order. FinalBucketSampler groups tones, not durations, so
        # reorder every input tied to the OTC loss together inside the batch.
        order = torch.argsort(input_lengths, descending=True)
        batch_log_probs = batch_log_probs.index_select(0, order)
        input_lengths = input_lengths.index_select(0, order)
        order_list = order.tolist()
        sequences = [sequences[index] for index in order_list]
        target_lengths = target_lengths.index_select(0, order)
        epoch = max(0, self.current_epoch)
        graph = self.otc_graph_compiler.compile(
            texts=[""] * len(sequences),
            word_ids_list=[[[phone] for phone in sequence] for sequence in sequences],
            allow_bypass_arc=self.args.allow_bypass_arc,
            allow_self_loop_arc=self.args.allow_self_loop_arc,
            bypass_weight=(self.args.initial_bypass_weight
                           * self.args.bypass_weight_decay ** epoch),
            self_loop_weight=(self.args.initial_self_loop_weight
                              * self.args.self_loop_weight_decay ** epoch),
        ).to(log_probs.device)
        supervision_segments = torch.stack((
            torch.arange(log_probs.shape[1], dtype=torch.int32),
            torch.zeros(log_probs.shape[1], dtype=torch.int32),
            input_lengths.detach().to(device="cpu", dtype=torch.int32),
        ), dim=1)
        dense_fsa = k2.DenseFsaVec(
            otc_log_probs, supervision_segments, allow_truncate=3)
        return k2.ctc_loss(
            decoding_graph=graph,
            dense_fsa_vec=dense_fsa,
            target_lengths=target_lengths.to(dtype=torch.int32),
            output_beam=self.args.otc_beam_size,
            reduction="mean",
            use_double_scores=True,
        )


def _yaml_argument_defaults(config, valid_keys):
    """Collect argparse-compatible leaves from the four YAML sections."""
    defaults = {}

    def visit(value):
        if not isinstance(value, dict):
            return
        for key, child in value.items():
            if key in valid_keys and not isinstance(child, (dict, list)):
                if key in defaults and defaults[key] != child:
                    raise ValueError(
                        f"Conflicting values for {key!r}: "
                        f"{defaults[key]!r} and {child!r}"
                    )
                defaults[key] = child
            visit(child)

    for section in ("data", "model", "training", "contrastive_learning"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"YAML requires a mapping section named {section!r}")
        visit(config[section])
    return defaults


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", default="configs/wav2vec2_otc_finalbucket_f0.yaml")
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=config_args.config,
                        help="Authoritative four-section YAML configuration")
    parser.add_argument("--train-path")
    parser.add_argument("--val-path")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--token-table-path", default="data/lang_jsonl/tokens.txt")
    parser.add_argument("--model-name", default="TencentGameMate/chinese-wav2vec2-large")
    parser.add_argument("--ctc-vocab", type=int, default=188)
    parser.add_argument("--otc-token-table-path", default="data/lang_jsonl/tokens_otc.txt")
    parser.add_argument("--otc-token", default="<star>")
    parser.add_argument("--otc-beam-size", type=float, default=10.0)
    parser.add_argument("--initial-bypass-weight", type=float, default=0.0)
    parser.add_argument("--initial-self-loop-weight", type=float, default=0.0)
    parser.add_argument("--bypass-weight-decay", type=float, default=1.0)
    parser.add_argument("--self-loop-weight-decay", type=float, default=1.0)
    parser.add_argument("--allow-bypass-arc", type=str2bool, default=True)
    parser.add_argument("--allow-self-loop-arc", type=str2bool, default=True)
    parser.add_argument("--exp-dir", default="exp/wav2vec2_otc_finalbucket_f0")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=3)
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument(
        "--strategy", default="ddp_find_unused_parameters_true",
        help="Wav2Vec2 LayerDrop can leave encoder-layer parameters unused on a step",
    )
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--freeze-backbone-steps", type=int, default=10000)
    parser.add_argument("--backbone-ramp-steps", type=int, default=3000)
    parser.add_argument("--head-warmup-steps", type=int, default=3000)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--mask-time-prob", type=float, default=0.65)
    parser.add_argument("--mask-time-length", type=int, default=10)
    parser.add_argument("--mask-feature-prob", type=float, default=0.25)
    parser.add_argument("--mask-feature-length", type=int, default=64)
    parser.add_argument("--f0-cache-dir", default=None)
    parser.add_argument("--f0-dim", type=int, default=256)
    parser.add_argument("--f0-normalization-enabled", type=str2bool, default=True)
    parser.add_argument("--f0-interpolation-enabled", type=str2bool, default=True)
    parser.add_argument("--f0-loss-weight", type=float, default=0.05)
    parser.add_argument("--final-bucket-sampling", action="store_true")
    parser.add_argument("--mfa-alignment-dir", default=None)
    parser.add_argument("--tone-acoustic-only", action="store_true")
    parser.add_argument("--tone-contrastive", action="store_true")
    parser.add_argument("--tone-loss-weight", type=float, default=0.05)
    parser.add_argument("--tone-temperature", type=float, default=0.1)
    parser.add_argument("--tone-start-step", type=int, default=10000)
    parser.add_argument("--tone-ramp-steps", type=int, default=2000)
    parser.add_argument("--tone-cross-accent-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)

    config_path = Path(config_args.config)
    if not config_path.is_file():
        parser.error(f"training config not found: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        parser.error("training config must contain a YAML mapping")
    valid_keys = {action.dest for action in parser._actions}
    try:
        parser.set_defaults(**_yaml_argument_defaults(config, valid_keys))
    except ValueError as exc:
        parser.error(str(exc))
    f0_config = config["data"]["f0"]
    parser.set_defaults(
        f0_normalization_enabled=f0_config["normalization"]["enabled"],
        f0_interpolation_enabled=f0_config["interpolation"]["enabled"],
    )

    args = parser.parse_args()
    if not args.train_path or not args.val_path:
        parser.error("train_path and val_path must be set in YAML or on the command line")
    if args.tone_temperature <= 0 or args.tone_cross_accent_weight < 1:
        parser.error("tone temperature must be > 0 and cross-accent weight must be >= 1")
    if args.f0_loss_weight < 0:
        parser.error("F0 loss weight must be non-negative")
    if args.f0_loss_weight > 0 and args.f0_cache_dir is None:
        parser.error("positive F0 loss weight requires --f0-cache-dir")
    if not Path(args.otc_token_table_path).is_file():
        parser.error(f"OTC token table not found: {args.otc_token_table_path}")
    if min(args.bypass_weight_decay, args.self_loop_weight_decay) < 0:
        parser.error("OTC weight decays must be non-negative")
    return args


def save_training_config(args):
    """Save both the authored YAML and effective flat runtime parameters."""
    config_dir = Path(args.exp_dir) / "checkpoints"
    config_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(args.config).resolve(), config_dir / "training_config.yaml")
    with (config_dir / "training_config.resolved.yaml").open(
            "w", encoding="utf-8") as stream:
        yaml.safe_dump(vars(args), stream, sort_keys=True, allow_unicode=True)


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)
    Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank == 0:
        save_training_config(args)

    use_f0 = args.f0_cache_dir is not None
    collator = Wav2Vec2Collator(args.model_name, use_f0=use_f0)
    train_dataset = QwenCTCDataset(
        args.train_path, args.data_root, f0_cache_dir=args.f0_cache_dir,
        f0_normalization_enabled=args.f0_normalization_enabled,
        f0_interpolation_enabled=args.f0_interpolation_enabled,
    )
    if args.mfa_alignment_dir:
        coverage = attach_mfa_intervals(train_dataset.rows, args.mfa_alignment_dir)
        train_dataset.rows = [r for r in train_dataset.rows if r["mfa_intervals"] is not None]
        coverage["retained"] = len(train_dataset.rows)
        coverage["dropped"] = coverage["total"] - coverage["retained"]
        if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0:
            (Path(args.exp_dir) / "mfa_coverage.json").write_text(
                json.dumps(coverage, indent=2) + "\n", encoding="utf-8"
            )

    sampler = FinalBucketSampler(
        train_dataset.rows, args.batch_size,
        world_size=args.devices if torch.cuda.is_available() else 1,
        seed=args.seed,
    ) if args.final_bucket_sampling else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        shuffle=sampler is None, drop_last=True, num_workers=args.workers,
        collate_fn=collator, persistent_workers=args.workers > 0,
    )

    val_dataset = QwenCTCDataset(
        args.val_path, args.data_root, f0_cache_dir=args.f0_cache_dir,
        f0_normalization_enabled=args.f0_normalization_enabled,
        f0_interpolation_enabled=args.f0_interpolation_enabled,
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=args.devices,
        rank=int(os.environ.get("LOCAL_RANK", "0")), shuffle=False,
    ) if args.final_bucket_sampling and torch.cuda.is_available() else None
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.workers, collate_fn=collator,
        persistent_workers=args.workers > 0,
    )

    checkpoint = ModelCheckpoint(
        dirpath=f"{args.exp_dir}/checkpoints",
        filename="checkpoint-step={step:06d}-per={val/per:.4f}",
        monitor="val/per", mode="min", save_top_k=2, save_last=True,
        auto_insert_metric_name=False, save_on_train_epoch_end=False,
    )
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices if torch.cuda.is_available() else 1,
        strategy=args.strategy if torch.cuda.is_available() else "auto",
        precision=args.precision, max_steps=args.max_steps, max_epochs=-1,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=1.0, gradient_clip_algorithm="norm",
        use_distributed_sampler=not args.final_bucket_sampling,
        limit_train_batches=(1.0 if args.final_bucket_sampling else
                             args.eval_steps * args.gradient_accumulation_steps),
        val_check_interval=(args.eval_steps * args.gradient_accumulation_steps
                            if args.final_bucket_sampling else 1.0),
        check_val_every_n_epoch=None if args.final_bucket_sampling else 1,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step"),
                   GracefulCheckpoint(f"{args.exp_dir}/checkpoints/last.ckpt")],
        logger=TensorBoardLogger(args.exp_dir, name="logs"),
    )
    trainer.fit(
        Wav2Vec2CTCModule(args), train_loader, val_loader,
        ckpt_path=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()

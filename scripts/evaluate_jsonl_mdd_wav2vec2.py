#!/usr/bin/env python3
"""Evaluate a Wav2Vec2 final-bucket/F0 CTC checkpoint on MDD manifests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from scripts.baseline.finetuning_pinyin_ctc_qwen3 import QwenCTCDataset
from scripts.baseline.finetuning_pinyin_ctc_wav2vec2 import (
    Wav2Vec2Collator,
    Wav2Vec2CTCModule,
)
from scripts.evaluate_jsonl_mdd_qwen3 import score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--data-root", default="/data2/xintong/mandarin_accent")
    parser.add_argument("--output-dir", default="results/wav2vec2_finalbucket_f0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model_args = argparse.Namespace(**checkpoint["hyper_parameters"])
    model = Wav2Vec2CTCModule(model_args)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    use_f0 = model_args.f0_cache_dir is not None
    collator = Wav2Vec2Collator(model_args.model_name, use_f0=use_f0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}

    for manifest in args.manifests:
        dataset = QwenCTCDataset(
            manifest, args.data_root, f0_cache_dir=model_args.f0_cache_dir
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            collate_fn=collator, pin_memory=device.type == "cuda",
        )
        predictions = []
        with torch.inference_mode():
            for batch in loader:
                rows, input_values, attention_mask, *optional = batch
                input_values = input_values.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                with torch.autocast(
                    "cuda", dtype=torch.float16, enabled=device.type == "cuda"
                ):
                    hidden, lengths, _, _ = model.encode_with_f0(
                        input_values, attention_mask,
                        optional[0] if optional else None,
                    )
                    logits = model.ctc_head(hidden)

                for row, sequence, length in zip(
                    rows, logits.argmax(-1).cpu(), lengths.cpu()
                ):
                    output, previous = [], -1
                    for token in sequence[:int(length)]:
                        token = int(token)
                        if token != previous and token != 0:
                            output.append(model.phones.id_to_phone[token])
                        previous = token
                    row["predicted_phones"] = output
                    predictions.append(row)

        name = Path(manifest).stem
        with (output_dir / f"{name}_predictions.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for row in predictions:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        summaries[name] = score(predictions)
        print(name, json.dumps(summaries[name], ensure_ascii=False), flush=True)

    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

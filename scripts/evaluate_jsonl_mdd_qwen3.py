#!/usr/bin/env python3
"""Evaluate a Qwen3 audio-encoder CTC checkpoint on paired MDD manifests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from scripts.baseline.finetuning_pinyin_ctc_qwen3 import (
    Qwen3CTCModule,
    QwenCollator,
    QwenCTCDataset,
)


def edit_distance(ref, hyp):
    previous = list(range(len(hyp) + 1))
    for i, ref_phone in enumerate(ref, 1):
        current = [i]
        for j, hyp_phone in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ref_phone != hyp_phone)))
        previous = current
    return previous[-1]


def align_to_canonical(canonical, predicted):
    n, m = len(canonical), len(predicted)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i - 1][j - 1] + (canonical[i - 1] != predicted[j - 1]),
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )
    aligned = [None] * n
    i, j = n, m
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (canonical[i - 1] != predicted[j - 1]):
            aligned[i - 1] = predicted[j - 1]
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return aligned


def score(rows):
    counts = {key: 0 for key in ("TA", "FA", "FN", "TN", "CD")}
    errors = reference_phones = 0
    for row in rows:
        canonical = [p for p in row["canonical_phones"] if p != "sil"]
        actual = [p for p in row["actual_phones"] if p != "sil"]
        actual_for_per = [p for p in actual if p is not None]
        labels = [v for p, v in zip(row["canonical_phones"], row["error_labels"]) if p != "sil"]
        predicted = row["predicted_phones"]
        errors += edit_distance(actual_for_per, predicted)
        reference_phones += len(actual_for_per)
        aligned = align_to_canonical(canonical, predicted)
        for can, act, label, pred in zip(canonical, actual, labels, aligned):
            is_error = bool(label) or act != can
            rejected = pred != can
            if not is_error and not rejected:
                counts["TA"] += 1
            elif is_error and not rejected:
                counts["FA"] += 1
            elif not is_error and rejected:
                counts["FN"] += 1
            else:
                counts["TN"] += 1
                if pred == act:
                    counts["CD"] += 1

    ta, fa, fn, tn, cd = (counts[k] for k in ("TA", "FA", "FN", "TN", "CD"))
    ratio = lambda a, b: 100.0 * a / b if b else 0.0
    precision = ratio(tn, tn + fn)
    recall = ratio(tn, tn + fa)
    return {
        "PER": ratio(errors, reference_phones),
        "FRR": ratio(fn, ta + fn),
        "FAR": ratio(fa, fa + tn),
        "Precision": precision,
        "Recall": recall,
        "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "DER": ratio(tn - cd, tn),
        **counts,
        "phone_errors": errors,
        "reference_phones": reference_phones,
        "utterances": len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--data-root", default="/data2/xintong/datasets/datasets")
    parser.add_argument("--output-dir", default="results/qwen3_asr_ctc")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model_args = argparse.Namespace(**checkpoint["hyper_parameters"])
    model = Qwen3CTCModule(model_args)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    use_f0 = getattr(model_args, "f0_cache_dir", None) is not None
    collator = QwenCollator(model_args.model_name, use_f0=use_f0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    special_ids = {0, 1, 2}
    for manifest in args.manifests:
        dataset = QwenCTCDataset(
            manifest, args.data_root,
            f0_cache_dir=getattr(model_args, "f0_cache_dir", None),
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            collate_fn=collator, pin_memory=device.type == "cuda",
        )
        predictions = []
        with torch.inference_mode():
            for batch in loader:
                rows, features, mask, *optional = batch
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    hidden, lengths, _, _ = model.encode_with_f0(
                        features.to(device, non_blocking=True),
                        mask.to(device, non_blocking=True),
                        optional[0] if optional else None,
                    )
                    logits = model.ctc_head(hidden)
                for row, sequence, length in zip(rows, logits.argmax(-1).cpu(), lengths.cpu()):
                    output, previous = [], -1
                    for token in sequence[:int(length)]:
                        token = int(token)
                        if token != previous and token not in special_ids:
                            output.append(model.phones.id_to_phone[token])
                        previous = token
                    row["predicted_phones"] = output
                    predictions.append(row)

        name = Path(manifest).stem
        prediction_path = output_dir / f"{name}_predictions.jsonl"
        with prediction_path.open("w", encoding="utf-8") as stream:
            for row in predictions:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        summaries[name] = score(predictions)
        print(name, json.dumps(summaries[name], ensure_ascii=False), flush=True)

    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

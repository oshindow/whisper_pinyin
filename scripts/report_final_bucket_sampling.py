#!/usr/bin/env python3
"""Audit FinalBucketSampler batches using the exact acoustic-only loss rules."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.baseline.final_bucket_sampler import FinalBucketSampler


def speaker(row):
    value = row.get("speaker_id")
    return ((str(row.get("source", "")), str(value))
            if value is not None and str(value) else None)


def accent(row):
    value = row.get("accent_id")
    return str(value) if value is not None and str(value) else None


def occurrences(row):
    return [(position, phone) for position, phone in enumerate(row["actual_phones"])
            if phone and phone[-1:] in "12345"]


def candidate_record(batch_rows, utterance_index, position, phone, temperature):
    anchor = batch_rows[utterance_index]
    anchor_speaker = speaker(anchor)
    positives, negatives = [], []
    for other_utterance, row in enumerate(batch_rows):
        for other_position, other_phone in occurrences(row):
            item = {
                "phone": other_phone,
                "utterance": other_utterance,
                "position": other_position,
                "id": row.get("id"),
                "source": row.get("source"),
                "speaker_id": row.get("speaker_id"),
                "accent_id": row.get("accent_id"),
                "cross_accent": accent(anchor) is not None and accent(row) is not None
                                and accent(anchor) != accent(row),
            }
            if other_phone == phone and anchor_speaker is not None \
                    and speaker(row) is not None and speaker(row) != anchor_speaker:
                positives.append(item)
            elif other_phone != phone and other_phone[:-1] == phone[:-1]:
                negatives.append(item)
    valid = bool(positives and negatives)
    positive_weight = 1.0 / len(positives) if valid else 0.0
    for item in positives:
        item["target_weight"] = positive_weight
        item["logit_scale"] = 1.0 / temperature
    for item in negatives:
        item["target_weight"] = 0.0
        item["logit_scale"] = 1.0 / temperature
    return {
        "anchor_phone": phone,
        "anchor_utterance": utterance_index,
        "anchor_position": position,
        "anchor_id": anchor.get("id"),
        "anchor_source": anchor.get("source"),
        "anchor_speaker_id": anchor.get("speaker_id"),
        "anchor_accent_id": anchor.get("accent_id"),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "valid_for_loss": valid,
        "positive_target_weight_each": positive_weight,
        "temperature": temperature,
        "logit_scale": 1.0 / temperature,
        "cross_accent_loss_multiplier": 1.0,
        "positives": positives,
        "negatives": negatives,
    }


def summarize(records):
    by_phone = {}
    for phone in sorted({record["anchor_phone"] for record in records}):
        selected = [record for record in records if record["anchor_phone"] == phone]
        valid = [record for record in selected if record["valid_for_loss"]]
        by_phone[phone] = {
            "anchors": len(selected),
            "valid_anchors": len(valid),
            "valid_fraction": len(valid) / len(selected),
            "positive_total": sum(r["positive_count"] for r in selected),
            "negative_total": sum(r["negative_count"] for r in selected),
            "positive_mean": sum(r["positive_count"] for r in selected) / len(selected),
            "negative_mean": sum(r["negative_count"] for r in selected) / len(selected),
        }
    return by_phone


def overall(records):
    valid = [record for record in records if record["valid_for_loss"]]
    return {
        "anchors": len(records),
        "valid_anchors": len(valid),
        "valid_fraction": len(valid) / max(1, len(records)),
        "positive_total": sum(r["positive_count"] for r in records),
        "negative_total": sum(r["negative_count"] for r in records),
        "positive_mean_all_anchors": sum(r["positive_count"] for r in records) / max(1, len(records)),
        "negative_mean_all_anchors": sum(r["negative_count"] for r in records) / max(1, len(records)),
        "positive_mean_valid_anchors": sum(r["positive_count"] for r in valid) / max(1, len(valid)),
        "negative_mean_valid_anchors": sum(r["negative_count"] for r in valid) / max(1, len(valid)),
    }


def markdown_candidate(item):
    cross = "cross" if item["cross_accent"] else "same"
    return (f'{item["phone"]} @ {item["source"]}/spk={item["speaker_id"]}'
            f'/accent={item["accent_id"]} ({cross}, w={item["target_weight"]:.4g})')


def write_markdown(path, report):
    lines = [
        "# Final-bucket sampling report", "",
        "## Configuration", "",
        f'- Manifest: `{report["configuration"]["manifest"]}`',
        f'- Seed / epoch: `{report["configuration"]["seed"]}` / `{report["configuration"]["epoch"]}`',
        f'- Single-GPU batch size: `{report["configuration"]["batch_size"]}`',
        f'- Temperature: `{report["configuration"]["temperature"]}` '
        f'(logit scale `{1 / report["configuration"]["temperature"]}`)', "",
        "Positive means the same phone from a different `(source, speaker_id)`. "
        "Negative means the same base final with a different tone. An anchor contributes "
        "only when both sets are non-empty. Positive target weights are `1 / P`; negatives "
        "have target weight 0 and occur only in the softmax denominator. Cross-accent loss "
        "multiplier is 1.0 in acoustic-only mode.", "",
        "## Overall", "",
        "| Scope | Anchors | Valid | Valid % | Pos./valid anchor | Neg./valid anchor |", "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stats in (("Full epoch", report["epoch_overall"]),
                        ("Sampled 10 batches", report["sampled_overall"])):
        lines.append(f'| {name} | {stats["anchors"]} | {stats["valid_anchors"]} | '
                     f'{100 * stats["valid_fraction"]:.2f}% | '
                     f'{stats["positive_mean_valid_anchors"]:.3f} | '
                     f'{stats["negative_mean_valid_anchors"]:.3f} |')
    lines += ["", "## Sampled batches", ""]
    for batch in report["sampled_batches"]:
        lines += [f'### Batch {batch["batch"]}', "",
                  "| Slot | ID | Source | Speaker | Accent | Phones |",
                  "|---:|---|---|---|---|---|"]
        for slot, row in enumerate(batch["utterances"]):
            phones = " ".join(str(phone) for phone in row["actual_phones"] if phone)
            lines.append(f'| {slot} | {row["id"]} | {row["source"]} | '
                         f'{row["speaker_id"]} | {row["accent_id"]} | {phones} |')
        lines += ["", "| Anchor | Source / speaker / accent | P | N | Valid | Positive candidates | Negative candidates |",
                  "|---|---|---:|---:|---|---|---|"]
        for anchor in batch["anchors"]:
            positives = "<br>".join(markdown_candidate(x) for x in anchor["positives"]) or "—"
            negatives = "<br>".join(markdown_candidate(x) for x in anchor["negatives"]) or "—"
            lines.append(
                f'| {anchor["anchor_phone"]} (pos={anchor["anchor_position"]}) | '
                f'{anchor["anchor_source"]} / {anchor["anchor_speaker_id"]} / '
                f'{anchor["anchor_accent_id"]} | {anchor["positive_count"]} | '
                f'{anchor["negative_count"]} | {"yes" if anchor["valid_for_loss"] else "no"} | '
                f'{positives} | {negatives} |')
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/jsonl_actual_split/train.jsonl")
    parser.add_argument("--output-dir", default="reports/final_bucket_sampling_seed42")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-sample-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    sampler = FinalBucketSampler(rows, args.batch_size, world_size=1, seed=args.seed)
    sampler.set_epoch(args.epoch)
    batches = sampler.batches()
    all_records, sample_records = [], []
    sample_batches = []
    for batch_number, indices in enumerate(batches):
        batch_rows = [rows[index] for index in indices]
        records = []
        for utterance_index, row in enumerate(batch_rows):
            for position, phone in occurrences(row):
                record = candidate_record(batch_rows, utterance_index, position, phone,
                                          args.temperature)
                record["batch"] = batch_number
                records.append(record)
        all_records.extend(records)
        if batch_number < args.num_sample_batches:
            sample_records.extend(records)
            sample_batches.append({
                "batch": batch_number,
                "indices": indices,
                "utterances": [{k: row.get(k) for k in
                                ("id", "source", "speaker_id", "accent_id", "duration",
                                 "actual_phones", "l2_wav")} for row in batch_rows],
                "anchors": records,
            })

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "configuration": vars(args),
        "semantics": {
            "positive": "same phone (final+tone), different (source, speaker_id)",
            "negative": "same base final, different tone; speaker is unrestricted",
            "valid_anchor": "requires at least one positive and one negative",
            "positive_target_weight": "1 / number of positives for that anchor",
            "negative_target_weight": 0.0,
            "logit": "cosine_similarity / temperature",
            "cross_accent_loss_multiplier": 1.0,
            "note": "tone_cross_accent_weight is not used in acoustic-only loss",
        },
        "dataset_rows": len(rows),
        "epoch_batches": len(batches),
        "epoch_anchor_occurrences": len(all_records),
        "epoch_valid_anchors": sum(r["valid_for_loss"] for r in all_records),
        "epoch_overall": overall(all_records),
        "epoch_by_phone": summarize(all_records),
        "sampled_batches": sample_batches,
        "sampled_overall": overall(sample_records),
        "sampled_by_phone": summarize(sample_records),
        "sampled_source_counts": dict(Counter(
            row["source"] for batch in sample_batches for row in batch["utterances"])),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(output / "sampled_anchors.md", report)
    with (output / "sampled_anchors.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["batch", "anchor_phone", "anchor_id", "anchor_source",
                  "anchor_speaker_id", "anchor_accent_id", "positive_count",
                  "negative_count", "valid_for_loss", "positive_target_weight_each",
                  "temperature", "logit_scale", "positives", "negatives"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in sample_records:
            row = {key: record[key] for key in fields}
            row["positives"] = json.dumps(row["positives"], ensure_ascii=False)
            row["negatives"] = json.dumps(row["negatives"], ensure_ascii=False)
            writer.writerow(row)
    print(json.dumps({key: report[key] for key in
                      ("dataset_rows", "epoch_batches", "epoch_anchor_occurrences",
                       "epoch_valid_anchors", "sampled_source_counts")},
                     ensure_ascii=False, indent=2))
    print(f"Wrote {output / 'report.json'}, {output / 'sampled_anchors.csv'} and "
          f"{output / 'sampled_anchors.md'}")


if __name__ == "__main__":
    main()

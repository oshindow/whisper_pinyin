#!/usr/bin/env python3
"""Filter JSONL audio by duration and make a source-stratified train/dev split."""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torchaudio

from preprocessing.preprocess_pinyin import _resolve_jsonl_audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("train.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("/data2/xintong/datasets/datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/jsonl_actual_split"))
    parser.add_argument("--dev-ratio", type=float, default=0.03)
    parser.add_argument("--min-duration", type=float, default=0.3)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    by_source = defaultdict(list)
    rejected = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            wav = _resolve_jsonl_audio(item["l2_wav"], args.data_root)
            try:
                info = torchaudio.info(str(wav))
                duration = info.num_frames / info.sample_rate
            except Exception as exc:
                rejected.append({"id": item.get("id"), "wav": str(wav), "reason": str(exc)})
                continue
            if not args.min_duration <= duration <= args.max_duration:
                rejected.append({"id": item.get("id"), "wav": str(wav), "duration": duration,
                                 "reason": "duration_out_of_range"})
                continue
            item["l2_wav"] = str(wav)
            item["duration"] = duration
            by_source[item.get("source", "unknown")].append(item)

    rng = random.Random(args.seed)
    train, dev = [], []
    source_counts = {}
    for source in sorted(by_source):
        rows = by_source[source]
        rng.shuffle(rows)
        n_dev = round(len(rows) * args.dev_ratio)
        if args.dev_ratio > 0 and rows:
            n_dev = max(1, n_dev)
        dev.extend(rows[:n_dev])
        train.extend(rows[n_dev:])
        source_counts[source] = {"total": len(rows), "train": len(rows) - n_dev, "dev": n_dev}

    rng.shuffle(train)
    rng.shuffle(dev)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("dev.jsonl", dev), ("rejected.jsonl", rejected)):
        with (args.output_dir / name).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "seed": args.seed, "dev_ratio": args.dev_ratio,
        "duration_range_seconds": [args.min_duration, args.max_duration],
        "train": len(train), "dev": len(dev), "rejected": len(rejected),
        "sources": source_counts,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

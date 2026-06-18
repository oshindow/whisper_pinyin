#!/usr/bin/env python3
"""Count AISHELL-3 speakers and utterances for train/test splits."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


SPEAKER_RE = re.compile(r"^(SSB\d{4})\d+")


def speaker_from_utt_id(utt_id: str) -> str:
    """Extract AISHELL-3 speaker id from an utterance id such as SSB00050001.wav."""
    stem = Path(utt_id).stem
    match = SPEAKER_RE.match(stem)
    if not match:
        raise ValueError(f"Cannot parse speaker id from utterance id: {utt_id}")
    return match.group(1)


def count_split(content_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with content_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            utt_id = line.split(maxsplit=1)[0]
            try:
                speaker = speaker_from_utt_id(utt_id)
            except ValueError as exc:
                raise ValueError(f"{content_path}:{line_no}: {exc}") from exc
            counts[speaker] += 1
    return counts


def load_excluded_speakers(spk_info_path: Path, excluded_accent: str) -> set[str]:
    excluded = set()
    if not spk_info_path.is_file():
        raise FileNotFoundError(f"Missing speaker info file: {spk_info_path}")

    with spk_info_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            speaker, _age, _gender, accent = parts[:4]
            if accent == excluded_accent:
                excluded.add(speaker)
    return excluded


def filter_counts(counts: Counter[str], excluded_speakers: set[str]) -> Counter[str]:
    return Counter(
        {speaker: count for speaker, count in counts.items() if speaker not in excluded_speakers}
    )


def sorted_counts(counts: Counter[str]) -> dict[str, int]:
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def print_split(name: str, counts: Counter[str]) -> None:
    print(f"\n[{name}]")
    print(f"speaker_total: {len(counts)}")
    print(f"utterance_total: {sum(counts.values())}")
    print(f"speakers_with_less_than_10_utterances: {sum(1 for count in counts.values() if count < 10)}")
    print(f"speakers_with_less_than_100_utterances: {sum(1 for count in counts.values() if count < 100)}")
    print("utterances_per_speaker:")
    for speaker, count in sorted_counts(counts).items():
        print(f"{speaker}\t{count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count train/test speaker totals, per-speaker utterance counts, "
            "and test speakers absent from train for AISHELL-3."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data2/xintong/aishell3"),
        help="AISHELL-3 root directory. Default: /data2/xintong/aishell3",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path to save the same statistics as JSON.",
    )
    parser.add_argument(
        "--exclude-accent",
        default="others",
        help="Exclude speakers with this accent in spk-info.txt. Default: others",
    )
    args = parser.parse_args()

    train_content = args.data_root / "train" / "content.txt"
    test_content = args.data_root / "test" / "content.txt"
    spk_info = args.data_root / "spk-info.txt"
    for path in (train_content, test_content):
        if not path.is_file():
            raise FileNotFoundError(f"Missing content file: {path}")

    excluded_speakers = load_excluded_speakers(spk_info, args.exclude_accent)
    train_counts = filter_counts(count_split(train_content), excluded_speakers)
    test_counts = filter_counts(count_split(test_content), excluded_speakers)
    seen_counts = Counter(
        {speaker: count for speaker, count in test_counts.items() if speaker in train_counts}
    )
    unseen_counts = Counter(
        {speaker: count for speaker, count in test_counts.items() if speaker not in train_counts}
    )

    print(f"data_root: {args.data_root}")
    print(f"excluded_accent: {args.exclude_accent}")
    print(f"excluded_speakers: {', '.join(sorted(excluded_speakers))}")
    print_split("train", train_counts)
    print_split("test", test_counts)
    print_split("test_seen_speakers_in_train", seen_counts)
    print_split("test_unseen_speakers_not_in_train", unseen_counts)

    summary = {
        "data_root": str(args.data_root),
        "excluded_accent": args.exclude_accent,
        "excluded_speakers": sorted(excluded_speakers),
        "train": {
            "speaker_total": len(train_counts),
            "utterance_total": sum(train_counts.values()),
            "speakers_with_less_than_10_utterances": sum(
                1 for count in train_counts.values() if count < 10
            ),
            "speakers_with_less_than_100_utterances": sum(
                1 for count in train_counts.values() if count < 100
            ),
            "utterances_per_speaker": sorted_counts(train_counts),
        },
        "test": {
            "speaker_total": len(test_counts),
            "utterance_total": sum(test_counts.values()),
            "speakers_with_less_than_10_utterances": sum(
                1 for count in test_counts.values() if count < 10
            ),
            "speakers_with_less_than_100_utterances": sum(
                1 for count in test_counts.values() if count < 100
            ),
            "utterances_per_speaker": sorted_counts(test_counts),
        },
        "test_seen_speakers_in_train": {
            "speaker_total": len(seen_counts),
            "utterance_total": sum(seen_counts.values()),
            "speakers_with_less_than_10_utterances": sum(
                1 for count in seen_counts.values() if count < 10
            ),
            "speakers_with_less_than_100_utterances": sum(
                1 for count in seen_counts.values() if count < 100
            ),
            "utterances_per_speaker": sorted_counts(seen_counts),
        },
        "test_unseen_speakers_not_in_train": {
            "speaker_total": len(unseen_counts),
            "utterance_total": sum(unseen_counts.values()),
            "speakers_with_less_than_10_utterances": sum(
                1 for count in unseen_counts.values() if count < 10
            ),
            "speakers_with_less_than_100_utterances": sum(
                1 for count in unseen_counts.values() if count < 100
            ),
            "utterances_per_speaker": sorted_counts(unseen_counts),
        },
    }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\njson_saved_to: {args.json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build an atomic-phone CTC lexicon and report missing JSONL audio files."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.phone_units import split_phone
from preprocessing.audio_paths import resolve_audio_path


SPECIAL_TOKENS = ("<blk>", "<sos/eos>", "<unk>")


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, default=Path("train.jsonl"))
    parser.add_argument(
        "--phone-field", choices=("actual_phones", "canonical_phones"),
        default="actual_phones", help="JSONL field used to build the CTC vocabulary.",
    )
    parser.add_argument(
        "--audit-jsonl", type=Path, nargs="*",
        default=[Path("train.jsonl"), Path("test_seen.jsonl"), Path("test_unseen.jsonl")],
    )
    parser.add_argument("--data-root", type=Path, default=Path("/data2/xintong/datasets/datasets"))
    parser.add_argument("--lexicon-dir", type=Path, default=Path("data/lang_jsonl"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/audio_audit"))
    parser.add_argument(
        "--unit", choices=("phone", "ift"), default="phone",
        help="phone: one CTC unit per annotated phone (tonal finals stay whole). "
             "ift: split every tonal final into a toneless final plus a tone, so "
             "one syllable becomes initial, final and tone.",
    )
    # Spelled out rather than argparse.BooleanOptionalAction, which the 3.8
    # training environment does not have.
    sil = parser.add_mutually_exclusive_group()
    sil.add_argument(
        "--exclude-sil", dest="exclude_sil", action="store_true",
        help="Exclude boundary sil from CTC vocabulary and lexicon (default).",
    )
    sil.add_argument(
        "--no-exclude-sil", dest="exclude_sil", action="store_false",
        help="Keep boundary sil as a CTC unit.",
    )
    parser.set_defaults(exclude_sil=True)
    args = parser.parse_args()

    phone_counts = Counter()
    train_rows = 0
    for item in read_jsonl(args.train_jsonl):
        train_rows += 1
        phones = item.get(args.phone_field)
        if not isinstance(phones, list):
            raise ValueError(f"{args.train_jsonl}: {item.get('id')} has no {args.phone_field} list")
        phone_counts.update(phone for phone in phones if phone is not None)

    if args.exclude_sil:
        phone_counts.pop("sil", None)
    phones = sorted(phone_counts)

    # The lexicon is keyed by the phones the manifests annotate, so splitting
    # tonal finals changes only the right-hand side and leaves the data alone.
    lexicon = {
        phone: split_phone(phone) if args.unit == "ift" else [phone]
        for phone in phones
    }
    units = sorted({unit for pieces in lexicon.values() for unit in pieces})

    args.lexicon_dir.mkdir(parents=True, exist_ok=True)
    (args.lexicon_dir / "lexicon.txt").write_text(
        "".join(f"{phone} {' '.join(lexicon[phone])}\n" for phone in phones), encoding="utf-8"
    )
    tokens = list(SPECIAL_TOKENS) + units
    (args.lexicon_dir / "tokens.txt").write_text(
        "".join(f"{token} {idx}\n" for idx, token in enumerate(tokens)), encoding="utf-8"
    )
    (args.lexicon_dir / "phone_counts.tsv").write_text(
        "phone\tcount\n" + "".join(f"{phone}\t{phone_counts[phone]}\n" for phone in phones),
        encoding="utf-8",
    )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for manifest in args.audit_jsonl:
        total = 0
        found = 0
        missing = []
        unseen_phones = Counter()
        for item in read_jsonl(manifest):
            total += 1
            raw_path = item.get("l2_wav", "")
            resolved = resolve_audio_path(raw_path, args.data_root)
            if raw_path and resolved.is_file():
                found += 1
            else:
                missing.append((str(item.get("id", "")), str(raw_path), str(resolved)))
            for phone in item.get(args.phone_field, []):
                if phone not in ("sil", None) and phone not in phone_counts:
                    unseen_phones[phone] += 1

        report = args.report_dir / f"{manifest.stem}_missing_wavs.tsv"
        report.write_text(
            "id\tmanifest_path\tresolved_path\n"
            + "".join("\t".join(row) + "\n" for row in missing),
            encoding="utf-8",
        )
        summary.append(
            {
                "manifest": str(manifest), "total": total, "found": found,
                "missing": len(missing), "missing_report": str(report),
                "phones_not_in_train": dict(sorted(unseen_phones.items())),
            }
        )

    summary_path = args.report_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "train_jsonl": str(args.train_jsonl), "train_rows": train_rows,
        "phone_field": args.phone_field, "unit": args.unit,
        "ctc_vocab_size": len(tokens), "phones": len(phones), "units": len(units),
        "exclude_sil": args.exclude_sil, "audits": summary,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(phones)} phones as {len(units)} {args.unit} units "
          f"({len(tokens)} tokens) to {args.lexicon_dir}")
    for item in summary:
        print(f"{item['manifest']}: found={item['found']}/{item['total']}, missing={item['missing']}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

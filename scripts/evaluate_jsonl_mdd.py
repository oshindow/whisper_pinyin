#!/usr/bin/env python3
"""Evaluate a Whisper-phone CTC checkpoint on JSONL MDD manifests."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
import torchaudio
from torch.utils.data import DataLoader, Dataset

import whisper
from preprocessing.phone_units import finals_from_lexicon, merge_units, read_lexicon
from preprocessing.preprocess_pinyin import _resolve_jsonl_audio


def edit_distance(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def align_to_canonical(canonical, predicted):
    """Return one predicted phone (or None) per canonical phone."""
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


class JsonlAudioDataset(Dataset):
    def __init__(self, manifest, data_root, rank=0, world_size=1):
        self.rows = []
        with open(manifest, encoding="utf-8") as f:
            for index, line in enumerate(f):
                if index % world_size != rank:
                    continue
                item = json.loads(line)
                item["resolved_wav"] = str(_resolve_jsonl_audio(item["l2_wav"], data_root))
                self.rows.append(item)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        item = self.rows[index]
        audio, sr = torchaudio.load(item["resolved_wav"])
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio.flatten()), n_mels=80)
        return item, mel


def collate(batch):
    rows, mels = zip(*batch)
    return list(rows), torch.stack(mels)


def load_tokens(path):
    table = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            symbol, idx = line.rsplit(maxsplit=1)
            table[int(idx)] = symbol
    return table


def ctc_decode(logits, tokens, finals=frozenset()):
    outputs = []
    for sequence in logits.argmax(-1).cpu().tolist():
        collapsed = []
        previous = None
        for token_id in sequence:
            if token_id != previous and token_id not in (0, 1, 2):
                collapsed.append(tokens[token_id])
            previous = token_id
        # Scoring stays at the phone level whatever the CTC units are, so an
        # initial/final/tone hypothesis is merged back into tonal finals here.
        outputs.append(merge_units(collapsed, finals))
    return outputs


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
    parser.add_argument("--tokens", default="data/lang_jsonl/tokens.txt")
    parser.add_argument(
        "--lexicon", default=None,
        help="Phone-to-unit lexicon; defaults to lexicon.txt beside --tokens. It tells "
             "which units a tone may attach to, and an identity lexicon disables merging.",
    )
    parser.add_argument("--ctc-vocab", type=int, default=188)
    parser.add_argument("--output-dir", default="results/whisper_pinyin_ctc_jsonl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    # Load on CPU first so checkpoint tensors and the constructed model are not
    # simultaneously resident on a busy GPU. FP16 is sufficient for inference.
    model = whisper.load_model(
        args.checkpoint, device="cpu", ctc_vocab=args.ctc_vocab, ctc_layers=2
    ).eval()
    if device.type == "cuda":
        model = model.half().to(device)
    tokens = load_tokens(args.tokens)
    lexicon_path = Path(args.lexicon) if args.lexicon else Path(args.tokens).parent / "lexicon.txt"
    finals = finals_from_lexicon(read_lexicon(lexicon_path)) if lexicon_path.is_file() else frozenset()
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for manifest in args.manifests:
        dataset = JsonlAudioDataset(manifest, args.data_root, rank, world_size)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=collate, pin_memory=True)
        local_rows = []
        with torch.inference_mode():
            for rows, mels in loader:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    features, _ = model.encoder(mels.to(device, non_blocking=True))
                    _, logits = model.ctc_head(features)
                for row, prediction in zip(rows, ctc_decode(logits, tokens, finals)):
                    row["predicted_phones"] = prediction
                    local_rows.append(row)

        gathered = [None] * world_size if rank == 0 else None
        if world_size > 1:
            dist.gather_object(local_rows, gathered, dst=0)
        else:
            gathered = [local_rows]
        if rank == 0:
            rows = sorted((row for shard in gathered for row in shard), key=lambda x: x["id"])
            name = Path(manifest).stem
            with (output_dir / f"{name}_predictions.jsonl").open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary = score(rows)
            all_summaries[name] = summary
            print(name, json.dumps(summary, ensure_ascii=False))

    if rank == 0:
        (output_dir / "summary.json").write_text(
            json.dumps(all_summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Visualize tone-conditioned Qwen3 encoder features using CTC forced alignment."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Subset

from scripts.baseline.finetuning_pinyin_ctc_qwen3 import (
    Qwen3CTCModule,
    QwenCollator,
    QwenCTCDataset,
)


TONES = "12345"


def choose_final(rows, phone_field):
    counts = Counter(
        phone for row in rows for phone in row[phone_field]
        if phone and phone[-1:] in TONES
    )
    grouped = defaultdict(dict)
    for phone, count in counts.items():
        grouped[phone[:-1]][phone[-1]] = count
    eligible = [(sum(value.values()), base, value) for base, value in grouped.items()
                if all(tone in value for tone in TONES)]
    if not eligible:
        raise ValueError("No final has examples for all five tones")
    _, base, tone_counts = max(eligible)
    return base, tone_counts


def ctc_viterbi(log_probs, targets, blank=0):
    """Return one extended-CTC-state index for every encoder frame."""
    target_length = len(targets)
    states = 2 * target_length + 1
    labels = torch.full((states,), blank, dtype=torch.long, device=log_probs.device)
    labels[1::2] = torch.as_tensor(targets, dtype=torch.long, device=log_probs.device)
    score = log_probs.new_full((states,), -torch.inf)
    score[0] = log_probs[0, blank]
    if states > 1:
        score[1] = log_probs[0, labels[1]]
    backpointers = []
    state_ids = torch.arange(states, device=log_probs.device)
    for frame in range(1, log_probs.shape[0]):
        candidates = torch.stack((
            score,
            torch.cat((score.new_full((1,), -torch.inf), score[:-1])),
            torch.cat((score.new_full((2,), -torch.inf), score[:-2])),
        ))
        # CTC cannot skip from one occurrence to the same repeated label.
        can_skip = torch.zeros(states, dtype=torch.bool, device=log_probs.device)
        can_skip[2:] = (labels[2:] != blank) & (labels[2:] != labels[:-2])
        candidates[2, ~can_skip] = -torch.inf
        best_score, transition = candidates.max(0)
        previous = state_ids - transition
        backpointers.append(previous)
        score = best_score + log_probs[frame, labels]
    state = states - 1 if states == 1 or score[-1] >= score[-2] else states - 2
    path = [state]
    for previous in reversed(backpointers):
        state = int(previous[state])
        path.append(state)
    path.reverse()
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/jsonl_actual_split/train.jsonl")
    parser.add_argument("--data-root", default="/data2/xintong/datasets/datasets")
    parser.add_argument("--audio-field", default="l2_wav",
                        help="Manifest field used as acoustic input (e.g. l2_wav or canonical_wav)")
    parser.add_argument("--audio-root", default=None,
                        help="Override acoustic input root, preserving l2_wav's path relative to /datasets/datasets/")
    parser.add_argument("--phone-field", default="actual_phones",
                        help="Phone sequence used for CTC alignment (e.g. actual_phones or canonical_phones)")
    parser.add_argument("--output-dir", default="results/qwen3_tone_tsne")
    parser.add_argument("--final", default=None, help="Final without tone; default: most frequent with tones 1--5")
    parser.add_argument("--samples-per-tone", type=int, default=100)
    parser.add_argument("--candidate-multiplier", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dataset = QwenCTCDataset(args.manifest, args.data_root, audio_field=args.audio_field)
    if args.audio_root:
        marker = "/datasets/datasets/"
        for row in dataset.rows:
            source = str(row["l2_wav"]).replace("\\", "/")
            if marker not in source:
                raise ValueError(f"Cannot derive relative audio path from l2_wav: {source}")
            row[args.audio_field] = str(Path(args.audio_root) / source.split(marker, 1)[1])
    selected_final, corpus_counts = choose_final(dataset.rows, args.phone_field)
    if args.final:
        selected_final = args.final
        corpus_counts = Counter(
            phone[-1] for row in dataset.rows for phone in row[args.phone_field]
            if phone in {selected_final + tone for tone in TONES}
        )
    target_phones = {selected_final + tone for tone in TONES}
    print("Selected final:", selected_final, "corpus counts:", dict(corpus_counts), flush=True)

    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    required_candidates = args.samples_per_tone * args.candidate_multiplier
    candidate_counts = Counter()
    selected_indices = []
    for index in indices:
        present = Counter(p for p in dataset.rows[index][args.phone_field] if p in target_phones)
        if any(candidate_counts[p[-1]] < required_candidates for p in present):
            selected_indices.append(index)
            for phone, count in present.items():
                candidate_counts[phone[-1]] += count
        if all(candidate_counts[tone] >= required_candidates for tone in TONES):
            break
    print("Candidate occurrences:", dict(candidate_counts), flush=True)
    missing_audio = [dataset.rows[index][args.audio_field] for index in selected_indices
                     if not Path(dataset.rows[index][args.audio_field]).is_file()]
    if missing_audio:
        examples = "\n  ".join(missing_audio[:5])
        raise FileNotFoundError(
            f"{len(missing_audio)}/{len(selected_indices)} selected {args.audio_field} files "
            f"do not exist. Examples:\n  {examples}"
        )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model_args = argparse.Namespace(**checkpoint["hyper_parameters"])
    model = Qwen3CTCModule(model_args)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(
        Subset(dataset, selected_indices), batch_size=args.batch_size,
        num_workers=args.num_workers, collate_fn=QwenCollator(model_args.model_name),
        pin_memory=device.type == "cuda",
    )

    vectors = {tone: [] for tone in TONES}
    metadata = {tone: [] for tone in TONES}
    with torch.inference_mode():
        for rows, features, mask in loader:
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                hidden, lengths = model.encode_hidden(features.to(device), mask.to(device))
                logits = model.ctc_head(hidden)
            for row, sequence_hidden, sequence_logits, length in zip(rows, hidden, logits, lengths):
                phones = [p for p in row[args.phone_field] if p not in (None, "sil")]
                if any(phone not in model.phones.phone_to_id for phone in phones):
                    continue
                target_ids = [model.phones.phone_to_id[p] for p in phones]
                log_probs = sequence_logits[:int(length)].float().log_softmax(-1)
                if len(target_ids) == 0 or int(length) < len(target_ids):
                    continue
                path = ctc_viterbi(log_probs, target_ids)
                state_to_frames = defaultdict(list)
                for frame, state in enumerate(path):
                    if state % 2:
                        state_to_frames[state // 2].append(frame)
                for target_index, phone in enumerate(phones):
                    if phone not in target_phones or len(vectors[phone[-1]]) >= args.samples_per_tone:
                        continue
                    frames = state_to_frames[target_index]
                    if not frames:
                        continue
                    vector = sequence_hidden[frames].float().mean(0).cpu().numpy()
                    vectors[phone[-1]].append(vector)
                    metadata[phone[-1]].append(
                        {"id": row.get("id"), "source": row.get("source"),
                         "audio": row.get(args.audio_field), "phone": phone,
                         "frames": len(frames)}
                    )
            if all(len(vectors[tone]) >= args.samples_per_tone for tone in TONES):
                break

    counts = {tone: len(vectors[tone]) for tone in TONES}
    if any(count < args.samples_per_tone for count in counts.values()):
        raise RuntimeError(f"Could not collect balanced samples: {counts}")
    x = np.concatenate([np.stack(vectors[tone]) for tone in TONES])
    labels = np.concatenate([[tone] * args.samples_per_tone for tone in TONES])
    # PCA denoising before t-SNE makes the result reproducible and faster.
    pca_dim = min(50, x.shape[0] - 1, x.shape[1])
    x_pca = PCA(n_components=pca_dim, random_state=args.seed).fit_transform(x)
    embedding = TSNE(
        n_components=2, perplexity=30, init="pca", learning_rate="auto",
        max_iter=2000, random_state=args.seed,
    ).fit_transform(x_pca)
    silhouette = float(silhouette_score(x_pca, labels))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "tone_latents_tsne.npz", latent=x, tsne=embedding, tone=labels)
    (output_dir / "metadata.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n"
                for tone in TONES for row in metadata[tone]), encoding="utf-8"
    )
    summary = {
        "checkpoint": args.checkpoint, "manifest": args.manifest,
        "audio_field": args.audio_field, "audio_root": args.audio_root,
        "phone_field": args.phone_field,
        "selected_final": selected_final, "corpus_counts": dict(corpus_counts),
        "samples_per_tone": args.samples_per_tone, "total_samples": len(labels),
        "pca_silhouette": silhouette, "seed": args.seed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
    plt.figure(figsize=(8, 6.5))
    for tone, color in zip(TONES, colors):
        keep = labels == tone
        plt.scatter(embedding[keep, 0], embedding[keep, 1], s=22, alpha=.72,
                    color=color, label=f"{selected_final}{tone} (n={keep.sum()})")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.title(f"Qwen3-ASR CTC-aligned latent features: {selected_final}1--{selected_final}5")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "tone_latents_tsne.png", dpi=300)
    plt.savefig(output_dir / "tone_latents_tsne.pdf")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

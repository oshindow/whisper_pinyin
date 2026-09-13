#!/usr/bin/env python3
"""CTC-aligned tone t-SNE for either branch of the paired Whisper-CTC model."""
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
import torchaudio
import whisper
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Dataset, Subset

TONES = "12345"


def read_tokens(path):
    table = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        symbol, index = line.rsplit(maxsplit=1)
        table[symbol] = int(index)
    return table


class AudioDataset(Dataset):
    def __init__(self, manifest, audio_field, audio_root=None):
        self.audio_field = audio_field
        self.rows = [json.loads(x) for x in Path(manifest).read_text(encoding="utf-8").splitlines() if x]
        if audio_root:
            marker = "/datasets/datasets/"
            for row in self.rows:
                source = row["l2_wav"].replace("\\", "/")
                if marker not in source:
                    raise ValueError(f"Cannot derive relative path from {source}")
                row[audio_field] = str(Path(audio_root) / source.split(marker, 1)[1])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio, sample_rate = torchaudio.load(row[self.audio_field])
        audio = audio.mean(0)
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)
        audio = audio[:30 * 16000]
        mel = whisper.log_mel_spectrogram(audio, n_mels=80)
        return row, mel, (mel.shape[-1] + 1) // 2


def collate(examples):
    if len(examples) != 1:
        raise ValueError("Unpadded Whisper inference requires --batch-size 1")
    rows, mels, lengths = zip(*examples)
    return list(rows), torch.stack(mels), torch.tensor(lengths)


def encode_unpadded(encoder, mel):
    """Whisper audio encoder with positional embeddings sliced to true length."""
    x = torch.nn.functional.gelu(encoder.conv1(mel))
    x_in = torch.nn.functional.gelu(encoder.conv2(x))
    x = x_in.permute(0, 2, 1)
    x = (x + encoder.positional_embedding[:x.shape[1]]).to(x.dtype)
    for block in encoder.blocks:
        x = block(x)
    return encoder.ln_post(x)


def choose_final(rows, phone_field):
    counts = Counter(p for row in rows for p in row[phone_field] if p and p[-1:] in TONES)
    grouped = defaultdict(dict)
    for phone, count in counts.items():
        grouped[phone[:-1]][phone[-1]] = count
    eligible = [(sum(x.values()), base, x) for base, x in grouped.items() if all(t in x for t in TONES)]
    if not eligible:
        raise ValueError("No final has all five tones")
    _, base, tone_counts = max(eligible)
    return base, tone_counts


def ctc_viterbi(log_probs, targets, blank=0):
    states = 2 * len(targets) + 1
    labels = torch.full((states,), blank, dtype=torch.long, device=log_probs.device)
    labels[1::2] = torch.as_tensor(targets, device=log_probs.device)
    score = log_probs.new_full((states,), -torch.inf)
    score[0] = log_probs[0, blank]
    if states > 1:
        score[1] = log_probs[0, labels[1]]
    backpointers = []
    state_ids = torch.arange(states, device=log_probs.device)
    for frame in range(1, log_probs.shape[0]):
        candidates = torch.stack((score, torch.cat((score.new_full((1,), -torch.inf), score[:-1])),
                                  torch.cat((score.new_full((2,), -torch.inf), score[:-2]))))
        can_skip = torch.zeros(states, dtype=torch.bool, device=log_probs.device)
        can_skip[2:] = (labels[2:] != blank) & (labels[2:] != labels[:-2])
        candidates[2, ~can_skip] = -torch.inf
        best, transition = candidates.max(0)
        backpointers.append(state_ids - transition)
        score = best + log_probs[frame, labels]
    state = states - 1 if states == 1 or score[-1] >= score[-2] else states - 2
    path = [state]
    for previous in reversed(backpointers):
        state = int(previous[state]); path.append(state)
    return path[::-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", default="data/jsonl_actual_paired_split/train.jsonl")
    p.add_argument("--tokens", default="data/lang_jsonl/tokens.txt")
    p.add_argument("--audio-field", choices=("l2_wav", "canonical_wav"), required=True)
    p.add_argument("--audio-root")
    p.add_argument("--phone-field", choices=("actual_phones", "canonical_phones"), required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--final", default="i")
    p.add_argument("--samples-per-tone", type=int, default=500)
    p.add_argument("--candidate-multiplier", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    dataset = AudioDataset(args.manifest, args.audio_field, args.audio_root)
    selected_final, corpus_counts = choose_final(dataset.rows, args.phone_field)
    if args.final:
        selected_final = args.final
        corpus_counts = Counter(p[-1] for row in dataset.rows for p in row[args.phone_field]
                                if p in {selected_final + t for t in TONES})
    targets = {selected_final + t for t in TONES}
    indices = list(range(len(dataset))); rng.shuffle(indices)
    required = args.samples_per_tone * args.candidate_multiplier
    candidate_counts, selected = Counter(), []
    for index in indices:
        present = Counter(p for p in dataset.rows[index][args.phone_field] if p in targets)
        if any(candidate_counts[p[-1]] < required for p in present):
            selected.append(index)
            for phone, count in present.items(): candidate_counts[phone[-1]] += count
        if all(candidate_counts[t] >= required for t in TONES): break
    missing = [dataset.rows[i][args.audio_field] for i in selected
               if not Path(dataset.rows[i][args.audio_field]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)}/{len(selected)} selected audio files missing; first: {missing[0]}")
    print("Selected final:", selected_final, "counts:", dict(corpus_counts),
          "candidates:", dict(candidate_counts), flush=True)
    # Put the rare fifth-tone candidates first so balanced collection stops early.
    selected.sort(key=lambda i: (selected_final + "5" not in dataset.rows[i][args.phone_field], i))

    model = whisper.load_model("medium", device="cpu", ctc_vocab=188, ctc_layers=2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    incompatible = model.load_state_dict(
        {k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()}, strict=False)
    unused_prefixes = ("decoder.", "stct_head.", "accent_classifier.", "acc_head.",
                       "accent_embedding.", "decoder_projection.")
    allowed_missing = tuple(k for k in incompatible.missing_keys if k.startswith(unused_prefixes))
    unexpected = tuple(incompatible.unexpected_keys)
    if len(allowed_missing) != len(incompatible.missing_keys) or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    model.decoder = None; model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); model.to(device)
    phone_to_id = read_tokens(args.tokens)
    loader = DataLoader(Subset(dataset, selected), batch_size=args.batch_size,
                        num_workers=args.num_workers, collate_fn=collate)
    vectors, metadata = {t: [] for t in TONES}, {t: [] for t in TONES}
    with torch.inference_mode():
        for batch_index, (rows, mels, lengths) in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                hidden = encode_unpadded(model.encoder, mels.to(device)); _, logits = model.ctc_head(hidden)
            for row, h, logit, length in zip(rows, hidden, logits, lengths):
                phones = [p for p in row[args.phone_field] if p not in (None, "sil")]
                if not phones or any(p not in phone_to_id for p in phones) or int(length) < len(phones): continue
                # Match training exactly: CTC supervision has one <sos/eos> at
                # each boundary. Phone k therefore occupies target state k + 1.
                boundary = phone_to_id["<sos/eos>"]
                alignment_ids = [boundary] + [phone_to_id[p] for p in phones] + [boundary]
                if int(length) < len(alignment_ids):
                    continue
                path = ctc_viterbi(logit[:int(length)].float().log_softmax(-1), alignment_ids)
                aligned = defaultdict(list)
                for frame, state in enumerate(path):
                    if state % 2: aligned[state // 2].append(frame)
                for target_index, phone in enumerate(phones):
                    tone = phone[-1:]
                    if phone not in targets or len(vectors[tone]) >= args.samples_per_tone: continue
                    frames = aligned[target_index + 1]
                    if not frames: continue
                    vectors[tone].append(h[frames].float().mean(0).cpu().numpy())
                    metadata[tone].append({"id": row.get("id"), "source": row.get("source"),
                                           "audio": row[args.audio_field], "phone": phone,
                                           "frames": len(frames)})
            if batch_index % 20 == 0:
                print("collected", {t: len(vectors[t]) for t in TONES}, flush=True)
            if all(len(vectors[t]) >= args.samples_per_tone for t in TONES): break

    counts = {t: len(vectors[t]) for t in TONES}
    if any(n < args.samples_per_tone for n in counts.values()):
        raise RuntimeError(f"Could not collect balanced samples: {counts}")
    x = np.concatenate([np.stack(vectors[t]) for t in TONES])
    labels = np.concatenate([[t] * args.samples_per_tone for t in TONES])
    x_pca = PCA(n_components=min(50, x.shape[0] - 1, x.shape[1]), random_state=args.seed).fit_transform(x)
    embedding = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
                     max_iter=2000, random_state=args.seed).fit_transform(x_pca)
    silhouette = float(silhouette_score(x_pca, labels))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "tone_latents_tsne.npz", latent=x, tsne=embedding, tone=labels)
    (out / "metadata.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
        for t in TONES for row in metadata[t]), encoding="utf-8")
    summary = vars(args) | {"selected_final": selected_final, "corpus_counts": dict(corpus_counts),
                            "total_samples": len(labels), "pca_silhouette": silhouette}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
    plt.figure(figsize=(8, 6.5))
    for tone, color in zip(TONES, colors):
        keep = labels == tone
        plt.scatter(embedding[keep, 0], embedding[keep, 1], s=22, alpha=.72, color=color,
                    label=f"{selected_final}{tone} (n={keep.sum()})")
    branch = "L2" if args.audio_field == "l2_wav" else "canonical"
    plt.xlabel("t-SNE dimension 1"); plt.ylabel("t-SNE dimension 2")
    plt.title(f"Paired Whisper-CTC {branch} latent features: {selected_final}1--{selected_final}5")
    plt.legend(frameon=False, ncol=2); plt.tight_layout()
    plt.savefig(out / "tone_latents_tsne.png", dpi=300); plt.savefig(out / "tone_latents_tsne.pdf")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

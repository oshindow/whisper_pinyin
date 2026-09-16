#!/usr/bin/env python3
"""Create hierarchy-aware final/tone plots from an existing t-SNE NPZ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


FINAL_COLORS = {
    "i": "#D55E00", "e": "#009E73", "u": "#4D4D4D",
    "ian": "#0072B2", "ai": "#CC79A7",
}
TONE_COLORS = {
    "1": "#D55E00", "2": "#E69F00", "3": "#009E73",
    "4": "#0072B2", "5": "#CC79A7",
}


def save_both(figure, output_dir, stem):
    figure.savefig(output_dir / f"{stem}.png", dpi=300)
    figure.savefig(output_dir / f"{stem}.pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--point-size", type=float, default=18)
    parser.add_argument("--alpha", type=float, default=0.62)
    args = parser.parse_args()

    output_dir = Path(args.input_dir)
    data = np.load(output_dir / "tone_latents_tsne.npz")
    latent, embedding = data["latent"], data["tsne"]
    phones = data["phone"].astype(str)
    finals = data["final"].astype(str)
    tones = data["tone"].astype(str)
    ordered_finals = list(dict.fromkeys(finals.tolist()))
    ordered_phones = [base + tone for base in ordered_finals for tone in "12345"
                      if np.any(phones == base + tone)]

    # A: global structure. Only final is encoded, so the hierarchy's top level is clear.
    figure, axis = plt.subplots(figsize=(10, 8))
    for index, base in enumerate(ordered_finals):
        keep = finals == base
        color = FINAL_COLORS.get(base, plt.get_cmap("tab10")(index))
        axis.scatter(embedding[keep, 0], embedding[keep, 1], s=args.point_size,
                     alpha=args.alpha, color=color, label=f"{base} (n={keep.sum()})")
    axis.set(xlabel="t-SNE dimension 1", ylabel="t-SNE dimension 2",
             title="Qwen3-ASR latent hierarchy — final level")
    axis.legend(frameon=False)
    figure.tight_layout()
    save_both(figure, output_dir, "hierarchy_A_global_finals")

    # B: one panel per final. Tone hue is consistent across panels and axes are shared.
    figure, axes = plt.subplots(1, len(ordered_finals), figsize=(7 * len(ordered_finals), 6),
                                sharex=True, sharey=True, squeeze=False)
    for axis, base in zip(axes[0], ordered_finals):
        for tone in "12345":
            keep = phones == base + tone
            if not keep.any():
                continue
            axis.scatter(embedding[keep, 0], embedding[keep, 1], s=args.point_size,
                         alpha=args.alpha, color=TONE_COLORS[tone],
                         label=f"tone {tone} (n={keep.sum()})")
        axis.set_title(f"final: {base}")
        axis.set_xlabel("t-SNE dimension 1")
        axis.legend(frameon=False, fontsize=9)
    axes[0, 0].set_ylabel("t-SNE dimension 2")
    figure.suptitle("Qwen3-ASR latent hierarchy — tone within each final")
    figure.tight_layout()
    save_both(figure, output_dir, "hierarchy_B_tone_facets")

    # C: mean pairwise cosine distance in original latent space. Unlike a
    # centroid-direction matrix, its diagonal exposes within-class dispersion.
    normalized = latent / np.linalg.norm(latent, axis=1, keepdims=True).clip(min=1e-12)
    distances = np.zeros((len(ordered_phones), len(ordered_phones)), dtype=np.float64)
    for row, phone in enumerate(ordered_phones):
        left = normalized[phones == phone]
        for column, other in enumerate(ordered_phones):
            right = normalized[phones == other]
            similarities = left @ right.T
            if row == column:
                similarities = similarities[~np.eye(len(left), dtype=bool)]
            distances[row, column] = 1.0 - similarities.mean()
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(distances, cmap="viridis", vmin=0, vmax=distances.max())
    axis.set_xticks(range(len(ordered_phones)), ordered_phones, rotation=45, ha="right")
    axis.set_yticks(range(len(ordered_phones)), ordered_phones)
    axis.set_title("Mean pairwise cosine distance (original latent space)")
    for row in range(len(ordered_phones)):
        for column in range(len(ordered_phones)):
            color = "white" if distances[row, column] > distances.max() * .55 else "black"
            axis.text(column, row, f"{distances[row, column]:.2f}",
                      ha="center", va="center", fontsize=7, color=color)
    figure.colorbar(image, ax=axis, label="cosine distance")
    figure.tight_layout()
    save_both(figure, output_dir, "hierarchy_C_pairwise_distance")

    # D: cross-final comparison. The main diagonal compares matching tones.
    if len(ordered_finals) == 2:
        left_phones = [ordered_finals[0] + tone for tone in "12345"]
        right_phones = [ordered_finals[1] + tone for tone in "12345"]
        left_indices = [ordered_phones.index(phone) for phone in left_phones]
        right_indices = [ordered_phones.index(phone) for phone in right_phones]
        cross = distances[np.ix_(left_indices, right_indices)]
        figure, axis = plt.subplots(figsize=(7.5, 6.5))
        image = axis.imshow(cross, cmap="viridis", vmin=distances.min(), vmax=distances.max())
        axis.set_xticks(range(5), right_phones)
        axis.set_yticks(range(5), left_phones)
        axis.set_xlabel(f"final {ordered_finals[1]}")
        axis.set_ylabel(f"final {ordered_finals[0]}")
        axis.set_title("Cross-final mean cosine distance\noutlined diagonal = same tone")
        for row in range(5):
            for column in range(5):
                color = "white" if cross[row, column] > distances.max() * .55 else "black"
                axis.text(column, row, f"{cross[row, column]:.2f}",
                          ha="center", va="center", fontsize=11, color=color)
            axis.add_patch(Rectangle((row - .5, row - .5), 1, 1, fill=False,
                                     edgecolor="white", linewidth=3))
        figure.colorbar(image, ax=axis, label="mean pairwise cosine distance")
        figure.tight_layout()
        save_both(figure, output_dir, "hierarchy_D_cross_final_tone_diagonal")

    pca_dim = min(50, len(latent) - 1, latent.shape[1])
    latent_pca = PCA(n_components=pca_dim, random_state=42).fit_transform(latent)
    metrics = {
        "samples": len(latent),
        "finals": ordered_finals,
        "phones": ordered_phones,
        "pca_final_silhouette": float(silhouette_score(latent_pca, finals)),
        "pca_phone_silhouette": float(silhouette_score(latent_pca, phones)),
        "pca_tone_silhouette_within_final": {
            base: float(silhouette_score(latent_pca[finals == base], tones[finals == base]))
            for base in ordered_finals
        },
        "pairwise_mean_cosine_distance": {
            phone: {other: float(distances[i, j]) for j, other in enumerate(ordered_phones)}
            for i, phone in enumerate(ordered_phones)
        },
        "within_class_mean_cosine_distance": {
            phone: float(distances[i, i]) for i, phone in enumerate(ordered_phones)
        },
    }
    if len(ordered_finals) == 2:
        off_diagonal = cross[~np.eye(5, dtype=bool)]
        metrics["cross_final_same_tone_distance_mean"] = float(np.diag(cross).mean())
        metrics["cross_final_different_tone_distance_mean"] = float(off_diagonal.mean())
        metrics["cross_final_same_vs_different_gap"] = float(
            off_diagonal.mean() - np.diag(cross).mean())
    (output_dir / "hierarchy_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in metrics
                      if key != "pairwise_mean_cosine_distance"},
                     indent=2))


if __name__ == "__main__":
    main()

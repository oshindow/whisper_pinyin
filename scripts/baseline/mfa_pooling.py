"""Read MFA phone intervals and pool Qwen's windowed CNN output frames."""
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch


def read_phone_intervals(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    for tier in re.split(r"\bitem\s*\[\d+\]\s*:", text)[1:]:
        if not re.search(r'\bname\s*=\s*"phones"', tier):
            continue
        intervals = []
        for block in re.split(r"\bintervals\s*\[\d+\]\s*:", tier)[1:]:
            start = float(re.search(r"\bxmin\s*=\s*([^\s]+)", block)[1])
            end = float(re.search(r"\bxmax\s*=\s*([^\s]+)", block)[1])
            phone = re.search(r'\btext\s*=\s*"((?:[^"]|"")*)"', block)[1].replace('""', '"')
            if phone in ("", "sil", "sp"):
                continue
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                raise ValueError("Invalid phone interval")
            if intervals and start < intervals[-1][2]:
                raise ValueError("Overlapping phone intervals")
            intervals.append((phone, start, end))
        duration = float(re.search(r"\bxmax\s*=\s*([^\s]+)", text)[1])
        return intervals, duration
    raise ValueError("TextGrid has no phones tier")


def attach_mfa_intervals(rows, root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"MFA alignment directory does not exist: {root}")
    indices = []
    # Prefer original recordings; target files are accepted only after the same
    # exact phone-sequence and audio-duration checks.
    directories = [root / "train_alignments", root / "target_alignments"]
    if not any(p.is_dir() for p in directories):
        directories = [root]
    for directory in directories:
        index = defaultdict(list)
        for path in directory.rglob("*.TextGrid"):
            stem = re.sub(r"^(?:a[13]_|target_\d+_)", "", path.stem)
            index[stem].append(path)
        indices.append(index)
    counts = Counter()
    by_source = defaultdict(Counter)
    for row in rows:
        expected = [p for p in row["actual_phones"] if p not in (None, "sil")]
        stem = Path(row["l2_wav"]).stem
        reason = "missing"
        row["mfa_intervals"] = None
        for index in indices:
            candidates = index.get(stem, [])
            if len(candidates) > 1:
                reason = "ambiguous"
                continue
            if not candidates:
                continue
            try:
                intervals, duration = read_phone_intervals(candidates[0])
            except (ValueError, TypeError, IndexError, OSError):
                reason = "invalid_textgrid"
                continue
            if [p for p, _, _ in intervals] != expected:
                reason = "phone_mismatch"
                continue
            if (not math.isfinite(duration) or "duration" not in row
                    or abs(duration - float(row["duration"])) > 0.03
                    or any(end > duration + 1e-6 for _, _, end in intervals)):
                reason = "duration_mismatch"
                continue
            row["mfa_intervals"] = [(start, end) for _, start, end in intervals]
            reason = "matched"
            break
        counts[reason] += 1
        by_source[row.get("source", "unknown")][reason] += 1
    summary = {"total": len(rows), "counts": dict(counts),
               "by_source": {source: dict(value) for source, value in by_source.items()}}
    print(f"MFA alignment coverage: {summary}", flush=True)
    if not counts["matched"]:
        raise ValueError("No MFA alignments match the training manifest")
    return summary


def interval_frame_indices(intervals, mel_length, chunk_size, hop_seconds):
    # Each of Qwen's three convolutions has k=3, stride=2, padding=1:
    # output centres are local mel indices 0, 8, 16, ... . The origin resets
    # at each n_window*2 chunk, including the partially filled final chunk.
    centres = [offset + local for offset in range(0, mel_length, chunk_size)
               for local in range(0, min(chunk_size, mel_length - offset), 8)]
    times = torch.tensor(centres, dtype=torch.float64) * hop_seconds
    result = []
    for start, end in intervals:
        frames = ((times >= start) & (times < end)).nonzero().flatten()
        if not len(frames) and len(times):
            frames = (times - (start + end) / 2).abs().argmin().reshape(1)
        result.append(frames.tolist())
    return result

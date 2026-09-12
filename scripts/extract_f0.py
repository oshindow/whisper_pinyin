#!/usr/bin/env python3
"""Pre-extract WORLD harvest F0 for every utterance referenced by JSONL manifests.

Harvest costs roughly a quarter of real time per core, so the tracks are
written to a cache that mirrors the corpus layout and the dataloader only has
to read them back.
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.f0_features import F0_CEIL, F0_FLOOR, FRAME_PERIOD_MS, extract_f0, f0_cache_path
from preprocessing.preprocess_pinyin import _resolve_jsonl_audio

_SETTINGS = {}


def _init_worker(settings):
    _SETTINGS.update(settings)


def _extract_one(audio_path):
    import torchaudio

    cache_path = f0_cache_path(audio_path, _SETTINGS["data_root"], _SETTINGS["cache_root"])
    if cache_path.is_file() and not _SETTINGS["overwrite"]:
        return audio_path, "skipped", 0
    try:
        audio, sample_rate = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(sample_rate, 16000)(audio)
        f0 = extract_f0(
            audio.flatten().numpy(), 16000,
            frame_period=_SETTINGS["frame_period"],
            f0_floor=_SETTINGS["f0_floor"], f0_ceil=_SETTINGS["f0_ceil"],
        )
    except Exception as exc:  # noqa: BLE001 - reported per utterance, never fatal
        return audio_path, f"failed: {exc}", 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, f0)
    return audio_path, "written", len(f0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--data-root", default="/data2/xintong/datasets/datasets")
    parser.add_argument("--cache-dir", default="/data2/xintong/f0_cache_world")
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--frame-period", type=float, default=FRAME_PERIOD_MS)
    parser.add_argument("--f0-floor", type=float, default=F0_FLOOR)
    parser.add_argument("--f0-ceil", type=float, default=F0_CEIL)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    # Fail once before starting workers instead of rejecting every utterance.
    try:
        import pyworld
    except ImportError as exc:
        parser.error(f"Cannot import pyworld: {exc}. Install F0 dependencies in "
                     f"{sys.executable} using requirements-f0.txt (see TWO_STAGE.md).")

    paths = []
    seen = set()
    for manifest in args.manifests:
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                path = str(_resolve_jsonl_audio(item["l2_wav"], args.data_root))
                if path not in seen:
                    seen.add(path)
                    paths.append(path)
    print(f"{len(paths)} unique utterances from {len(args.manifests)} manifest(s)")

    settings = {
        "data_root": args.data_root, "cache_root": args.cache_dir,
        "frame_period": args.frame_period, "f0_floor": args.f0_floor,
        "f0_ceil": args.f0_ceil, "overwrite": args.overwrite,
    }
    counts = {"written": 0, "skipped": 0, "failed": 0}
    failures = []
    started = time.time()
    with Pool(args.jobs, initializer=_init_worker, initargs=(settings,)) as pool:
        for index, (path, status, _) in enumerate(pool.imap_unordered(_extract_one, paths, chunksize=16), 1):
            if status.startswith("failed"):
                counts["failed"] += 1
                failures.append((path, status))
            else:
                counts[status] += 1
            if index % 2000 == 0 or index == len(paths):
                rate = index / max(1e-9, time.time() - started)
                remaining = (len(paths) - index) / max(1e-9, rate)
                print(f"{index}/{len(paths)}  {rate:.1f} utt/s  eta {remaining / 60:.1f} min", flush=True)

    print(f"written={counts['written']} skipped={counts['skipped']} failed={counts['failed']} "
          f"in {(time.time() - started) / 60:.1f} min")
    if failures:
        report = Path(args.cache_dir) / "failures.tsv"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("".join(f"{p}\t{s}\n" for p, s in failures), encoding="utf-8")
        print(f"failures listed in {report}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

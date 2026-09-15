"""Frame-level F0 features for the pinyin CTC recipes.

WORLD harvest is run at a 20 ms frame period, which is exactly the Whisper
encoder frame rate: the mel hop is 10 ms and the encoder's second convolution
halves it, so a 30 s chunk becomes 1500 encoder frames. F0 is extracted on the
unpadded waveform, which is far cheaper than padding every utterance to 30 s,
and the feature matrix is zero-padded to the encoder length when it is loaded.
"""

import hashlib
from pathlib import Path

import numpy as np

FRAME_PERIOD_MS = 20.0
F0_FLOOR = 60.0
F0_CEIL = 600.0
N_F0_FEATURES = 3


def extract_f0(audio, sample_rate, frame_period=FRAME_PERIOD_MS,
               f0_floor=F0_FLOOR, f0_ceil=F0_CEIL):
    """Run WORLD harvest plus stonemask refinement on one waveform."""
    import pyworld

    x = np.ascontiguousarray(np.asarray(audio, dtype=np.float64).reshape(-1))
    f0, times = pyworld.harvest(
        x, sample_rate, f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=frame_period
    )
    return pyworld.stonemask(x, f0, times, sample_rate).astype(np.float32)


def f0_to_features(f0, n_frames):
    """Return (3, n_frames): normalised log F0, its delta, and a voiced flag.

    Unvoiced frames are interpolated over so the delta stays meaningful across
    gaps, and the contour is normalised per utterance because tone is carried
    by relative pitch movement rather than by the speaker's absolute range.
    """
    f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
    features = np.zeros((N_F0_FEATURES, n_frames), dtype=np.float32)
    voiced = f0 > 0
    if not voiced.any():
        return features

    log_f0 = np.zeros_like(f0, dtype=np.float32)
    log_f0[voiced] = np.log(f0[voiced])
    index = np.arange(len(f0), dtype=np.float32)
    contour = np.interp(index, index[voiced], log_f0[voiced]).astype(np.float32)
    contour = (contour - log_f0[voiced].mean()) / (log_f0[voiced].std() + 1e-5)
    delta = np.gradient(contour).astype(np.float32) if len(contour) > 1 else np.zeros_like(contour)

    length = min(len(f0), n_frames)
    features[0, :length] = contour[:length]
    features[1, :length] = delta[:length]
    features[2, :length] = voiced[:length]
    return features


def f0_cache_path(audio_path, data_root, cache_root):
    """Mirror the corpus layout below cache_root so cache keys cannot collide."""
    audio_path = Path(audio_path)
    data_root = Path(data_root)
    cache_root = Path(cache_root)
    for candidate, root in ((audio_path, data_root),
                            (Path(str(audio_path)).resolve(), data_root.resolve())):
        try:
            return cache_root / candidate.relative_to(root).with_suffix(".npy")
        except (ValueError, OSError):
            continue
    # Audio outside the data root still needs a stable, unique key.
    digest = hashlib.sha1(str(audio_path).encode("utf-8")).hexdigest()
    return cache_root / "_external" / digest[:2] / f"{audio_path.stem}_{digest[:10]}.npy"


def load_f0_features(audio_path, data_root, cache_root, n_frames):
    """Load a cached F0 track and turn it into the frame-level feature matrix."""
    path = f0_cache_path(audio_path, data_root, cache_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing F0 cache for {audio_path} at {path}; run scripts/extract_f0.py first"
        )
    return f0_to_features(np.load(path), n_frames)

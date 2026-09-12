"""Resolve JSONL audio paths against the locally restored corpus layout."""
from pathlib import Path


def resolve_audio_path(raw_path, data_root):
    path = Path(raw_path)
    if path.is_file() or data_root is None:
        return path

    root = Path(data_root)
    marker = '/datasets/datasets/'
    normalized = str(path).replace('\\', '/')
    if marker not in normalized:
        return path
    relative = Path(normalized.split(marker, 1)[1])
    candidates = [root / relative]
    if relative.parts and relative.parts[0] == 'LATIC':
        candidates.append(root / 'magichub_multiaccent' / relative)
        # The sibling corpus is /data2/xintong/LATIC when the data root is
        # /data2/xintong/datasets/datasets. Its audio directory is named WAVA.
        corpus_roots = [root / 'LATIC', root / 'magichub_multiaccent' / 'LATIC']
        if root.name == 'datasets' and root.parent.name == 'datasets':
            corpus_roots.append(root.parent.parent / 'LATIC')
        suffix = Path(*relative.parts[1:])
        for corpus_root in corpus_roots:
            candidates.append(corpus_root / suffix)
            if suffix.parts and suffix.parts[0] == 'WAVE':
                candidates.append(corpus_root / 'WAVA' / Path(*suffix.parts[1:]))
    return next((candidate for candidate in candidates if candidate.is_file()), path)

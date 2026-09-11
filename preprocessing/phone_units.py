"""Initial / final / tone units for the pinyin CTC recipes.

The JSONL manifests annotate each syllable as an initial plus a tonal final
(``sh`` ``i1``). Splitting the tonal final into a toneless final and a tone
turns one syllable into three CTC units without touching the manifests: the
lexicon maps each annotated phone to its unit sequence and the training code
already expands lexicon entries into CTC targets.
"""

TONES = ("1", "2", "3", "4", "5")


def split_phone(phone):
    """Return the CTC units of one annotated phone.

    A tonal final becomes ``[toneless final, tone]``; every other phone,
    including the placeholder initials of zero-initial syllables, is kept whole.
    """
    if phone and phone[-1] in TONES:
        return [phone[:-1], phone[-1]]
    return [phone]


def read_lexicon(path):
    """Read a ``phone unit [unit ...]`` lexicon into a dict."""
    lexicon = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split()
            if fields:
                lexicon[fields[0]] = fields[1:]
    return lexicon


def finals_from_lexicon(lexicon):
    """Collect the units a tone may attach to.

    A unit counts as a final only because some lexicon entry ends with that
    unit followed by a tone, so an identity (tonal-final) lexicon yields an
    empty set and leaves :func:`merge_units` a no-op.
    """
    finals = set()
    for pieces in lexicon.values():
        if len(pieces) >= 2 and pieces[-1] in TONES:
            finals.add(pieces[-2])
    return finals


def merge_units(units, finals):
    """Merge tone units back onto the final they follow.

    Merging is decided by unit identity, never by position: a tone is attached
    only when it directly follows a final, so a final whose tone the model
    failed to emit stays a bare final and a tone whose final is missing stays a
    bare tone. Both then score as the errors they are instead of being stitched
    onto the wrong neighbour.
    """
    if not finals:
        return list(units)
    phones = []
    index = 0
    while index < len(units):
        unit = units[index]
        if unit in finals and index + 1 < len(units) and units[index + 1] in TONES:
            phones.append(unit + units[index + 1])
            index += 2
        else:
            phones.append(unit)
            index += 1
    return phones

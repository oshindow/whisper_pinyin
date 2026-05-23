import re
import torch
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO, Tuple, Union
import kaldialign
import k2
import random
from lhotse.dataset.signal_transforms import time_warp as time_warp_impl

Pathlike = Union[str, Path]

import re
def normlizer(text):
    return re.sub(r"[^\u4e00-\u9fa5]", "", text)

def make_pad_mask(
    lengths: torch.Tensor,
    max_len: int = 0,
    pad_left: bool = False,
) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
      pad_left:
        If ``False`` (default), padding is on the right.
        If ``True``, padding is on the left.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expanded_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    if pad_left:
        mask = expanded_lengths < (max_len - lengths).unsqueeze(1)
    else:
        mask = expanded_lengths >= lengths.unsqueeze(-1)

    return mask


def error_stats(
    f: TextIO,
    test_set_name: str,
    results: List[Tuple[str, str]],
    compute_CER: bool = True,
    sclite_mode: bool = False,
    enable_log: bool = False
) -> float:
    """Write statistics based on predicted results and reference transcripts.

    It will write the following to the given file:

        - WER
        - number of insertions, deletions, substitutions, corrects and total
          reference words. For example::

              Errors: 23 insertions, 57 deletions, 212 substitutions, over 2606
              reference words (2337 correct)

        - The difference between the reference transcript and predicted result.
          An instance is given below::

            THE ASSOCIATION OF (EDISON->ADDISON) ILLUMINATING COMPANIES

          The above example shows that the reference word is `EDISON`,
          but it is predicted to `ADDISON` (a substitution error).

          Another example is::

            FOR THE FIRST DAY (SIR->*) I THINK

          The reference word `SIR` is missing in the predicted
          results (a deletion error).
      results:
        An iterable of tuples. The first element is the cut_id, the second is
        the reference transcript and the third element is the predicted result.
      enable_log:
        If True, also print detailed WER to the console.
        Otherwise, it is written only to the given file.
    Returns:
      Return None.
    """
    subs: Dict[Tuple[str, str], int] = defaultdict(int)
    ins: Dict[str, int] = defaultdict(int)
    dels: Dict[str, int] = defaultdict(int)
    words: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    num_corr = 0
    ERR = "*"

    if compute_CER:
        results = [(cut_id, list(ref), list(hyp)) for cut_id, ref, hyp in results]


    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR, sclite_mode=sclite_mode)
        for ref_word, hyp_word in ali:
            if ref_word == ERR:
                ins[hyp_word] += 1
                words[hyp_word][3] += 1
            elif hyp_word == ERR:
                dels[ref_word] += 1
                words[ref_word][4] += 1
            elif hyp_word != ref_word:
                subs[(ref_word, hyp_word)] += 1
                words[ref_word][1] += 1
                words[hyp_word][2] += 1
            else:
                words[ref_word][0] += 1
                num_corr += 1
    ref_len = sum([len(r) for _, r, _ in results])
    sub_errs = sum(subs.values())
    ins_errs = sum(ins.values())
    del_errs = sum(dels.values())
    tot_errs = sub_errs + ins_errs + del_errs
    tot_err_rate = "%.2f" % (100.0 * tot_errs / ref_len if ref_len > 0 else 0.0)

    if enable_log:
      print(
            f"[{test_set_name}] %WER {tot_errs / ref_len:.2%} "
            f"[{tot_errs} / {ref_len}, {ins_errs} ins, "
            f"{del_errs} del, {sub_errs} sub ]"
        , file=f)
    
    if f is not None:
        print("PER-UTT DETAILS: corr or (ref->hyp)  ", file=f)

    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
        combine_successive_errors = True
        if combine_successive_errors:
            ali = [[[x], [y]] for x, y in ali]
            new_ali = []

            for i in range(len(ali) - 1):
                if ali[i][0] != ali[i][1] and ali[i + 1][0] != ali[i + 1][1]:
                    ali[i + 1][0] = ali[i][0] + ali[i + 1][0]
                    ali[i + 1][1] = ali[i][1] + ali[i + 1][1]
                else:
                    new_ali.append(ali[i])
            
            new_ali.append(ali[-1])  # Add last element
            ali = new_ali

            ali = [
                [
                    ERR if not x else " ".join(x),
                    ERR if not y else " ".join(y),
                ]
                for x, y in ali
            ]
        if f is not None:
            print(
                f"{cut_id}:\t"
                + " ".join(
                    (
                        ref_word if ref_word == hyp_word else f"({ref_word}->{hyp_word})"
                        for ref_word, hyp_word in ali
                    )
                ),
                file=f,
            )
    if f is not None:
        print("", file=f)
        print("SUBSTITUTIONS: count ref -> hyp", file=f)

        for count, (ref, hyp) in sorted([(v, k) for k, v in subs.items()], reverse=True):
            print(f"{count}   {ref} -> {hyp}", file=f)

        print("", file=f)
        print("DELETIONS: count ref", file=f)
        for count, ref in sorted([(v, k) for k, v in dels.items()], reverse=True):
            print(f"{count}   {ref}", file=f)

        print("", file=f)
        print("INSERTIONS: count hyp", file=f)
        for count, hyp in sorted([(v, k) for k, v in ins.items()], reverse=True):
            print(f"{count}   {hyp}", file=f)

        print("", file=f)
        print("PER-WORD STATS: word  corr tot_errs count_in_ref count_in_hyp", file=f)
    for _, word, counts in sorted(
        [(sum(v[1:]), k, v) for k, v in words.items()], reverse=True
    ):
        (corr, ref_sub, hyp_sub, ins, dels) = counts
        tot_errs = ref_sub + hyp_sub + ins + dels
        ref_count = corr + ref_sub + dels
        hyp_count = corr + hyp_sub + ins
        if f is not None:
            print(f"{word}   {corr} {tot_errs} {ref_count} {hyp_count}", file=f)
   
    return float(tot_err_rate)

def encode_supervisions_otc(
    supervisions: dict,
    subsampling_factor: int,
    token_ids: Optional[List[List[int]]] = None,
) -> Tuple[torch.Tensor, Union[List[str], List[List[int]]]]:
    """
    Encodes Lhotse's ``batch["supervisions"]`` dict into
    a pair of torch Tensor, and a list of transcription strings or token indexes

    The supervision tensor has shape ``(batch_size, 3)``.
    Its second dimension contains information about sequence index [0],
    start frames [1] and num frames [2].

    The batch items might become re-ordered during this operation -- the
    returned tensor and list of strings are guaranteed to be consistent with
    each other.
    """
    supervision_segments = torch.stack(
        (
            supervisions["sequence_idx"],
            torch.div(
                supervisions["start_frame"],
                subsampling_factor,
                rounding_mode="floor",
            ),
            torch.div(
                supervisions["num_frames"],
                subsampling_factor,
                rounding_mode="floor",
            ),
        ),
        1,
    ).to(torch.int32)

    indices = torch.argsort(supervision_segments[:, 2], descending=True)
    supervision_segments = supervision_segments[indices]

    ids = []
    verbatim_texts = []
    sorted_ids = []
    sorted_verbatim_texts = []
    pinyins = []
    for cut in supervisions["cut"]:
        id = cut.id
        if hasattr(cut.supervisions[0], "verbatim_text"):
            verbatim_text = cut.supervisions[0].verbatim_text
        else:
            verbatim_text = ""
        ids.append(id)
        verbatim_texts.append(verbatim_text)
    for cut in supervisions["cut"]:
        id = cut.id
        if hasattr(cut.supervisions[0], "custom"):
            pinyin = cut.supervisions[0].custom["pinyin"]
        else:
            pinyin = ""
        ids.append(id)
        pinyins.append(pinyin)

    for index in indices.tolist():
        sorted_ids.append(ids[index])
        sorted_verbatim_texts.append(verbatim_texts[index])

    if token_ids is None:
        texts = supervisions["text"]
        res = [texts[idx] for idx in indices]
    else:
        res = [token_ids[idx] for idx in indices]

    return supervision_segments, res, sorted_ids, sorted_verbatim_texts, pinyins

def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_texts(
    best_paths: k2.Fsa, return_ragged: bool = False
) -> Union[List[List[int]], k2.RaggedTensor]:
    """Extract the texts (as word IDs) from the best-path FSAs.
    Args:
      best_paths:
        A k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
        containing multiple FSAs, which is expected to be the result
        of k2.shortest_path (otherwise the returned values won't
        be meaningful).
      return_ragged:
        True to return a ragged tensor with two axes [utt][word_id].
        False to return a list-of-list word IDs.
    Returns:
      Returns a list of lists of int, containing the label sequences we
      decoded.
    """
    if isinstance(best_paths.aux_labels, k2.RaggedTensor):
        # remove 0's and -1's.
        aux_labels = best_paths.aux_labels.remove_values_leq(0)
        # TODO: change arcs.shape() to arcs.shape
        aux_shape = best_paths.arcs.shape().compose(aux_labels.shape)

        # remove the states and arcs axes.
        aux_shape = aux_shape.remove_axis(1)
        aux_shape = aux_shape.remove_axis(1)
        aux_labels = k2.RaggedTensor(aux_shape, aux_labels.values)
    else:
        # remove axis corresponding to states.
        aux_shape = best_paths.arcs.shape().remove_axis(1)
        aux_labels = k2.RaggedTensor(aux_shape, best_paths.aux_labels)
        # remove 0's and -1's.
        aux_labels = aux_labels.remove_values_leq(0)

    assert aux_labels.num_axes == 2
    if return_ragged:
        return aux_labels
    else:
        return aux_labels.tolist()
    
def time_warp(
    features: torch.Tensor,
    p: float = 0.9,
    time_warp_factor: Optional[int] = 80,
    supervision_segments: Optional[torch.Tensor] = None,
):
    """Apply time warping on a batch of features"""
    if time_warp_factor is None or time_warp_factor < 1:
        return features
    assert (
        len(features.shape) == 3
    ), f"SpecAugment only supports batches of single-channel feature matrices. {features.shape}"
    features = features.clone()
    if supervision_segments is None:
        # No supervisions - apply spec augment to full feature matrices.
        for sequence_idx in range(features.size(0)):
            if random.random() > p:
                # Randomly choose whether this transform is applied
                continue
            features[sequence_idx] = time_warp_impl(
                features[sequence_idx], factor=time_warp_factor
            )
    else:
        # Supervisions provided - we will apply time warping only on the supervised areas.
        for sequence_idx, start_frame, num_frames in supervision_segments:
            if random.random() > p:
                # Randomly choose whether this transform is applied
                continue
            end_frame = start_frame + num_frames
            features[sequence_idx, start_frame:end_frame] = time_warp_impl(
                features[sequence_idx, start_frame:end_frame], factor=time_warp_factor
            )

    return features
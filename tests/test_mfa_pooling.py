import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.baseline.mfa_pooling import attach_mfa_intervals, interval_frame_indices, read_phone_intervals


def textgrid(phones):
    blocks = '\n'.join(f'intervals [{i}]:\nxmin = {start}\nxmax = {end}\ntext = "{phone}"'
                       for i, (phone, start, end) in enumerate(phones, 1))
    return f'xmin = 0\nxmax = 1\nitem [1]:\nname = "phones"\n{blocks}\n'


class MFAPoolingTests(unittest.TestCase):
    def test_occurrences_and_target_fallback_require_matching_labels_and_duration(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            train, target = root / 'train_alignments', root / 'target_alignments'
            train.mkdir()
            target.mkdir()
            (train / 'a3_utt.TextGrid').write_text(textgrid([('i2', 0, 1)]))
            expected = [('i1', .1, .2), ('', .2, .3), ('i1', .3, .4), ('i2', .4, .9)]
            (target / 'target_0000000_utt.TextGrid').write_text(textgrid(expected))
            intervals, _ = read_phone_intervals(target / 'target_0000000_utt.TextGrid')
            self.assertEqual([p for p, _, _ in intervals], ['i1', 'i1', 'i2'])
            rows = [dict(l2_wav='utt.wav', actual_phones=['sil', 'i1', None, 'i1', 'i2'], duration=1),
                    dict(l2_wav='utt.wav', actual_phones=['i1', 'i1', 'i2'], duration=2),
                    dict(l2_wav='missing.wav', actual_phones=['i1'], duration=1)]
            counts = attach_mfa_intervals(rows, root)['counts']
            self.assertEqual(counts, {'matched': 1, 'duration_mismatch': 1, 'missing': 1})
            self.assertEqual(rows[0]['mfa_intervals'], [(.1, .2), (.3, .4), (.4, .9)])
            self.assertIsNone(rows[1]['mfa_intervals'])

    def test_chunk_origin_short_interval_and_gradient(self):
        # Chunk 0: .00,.08,...,.96; chunk 1: 1.00,1.08; not 1.04,1.12.
        frames = interval_frame_indices([(.95, 1.01), (1.02, 1.03), (1.07, 1.09)], 110, 100, .01)
        self.assertEqual(frames, [[12, 13], [13], [14]])
        hidden = torch.randn(15, 4, requires_grad=True)
        hidden[frames[0]].mean(0).sum().backward()
        self.assertTrue(torch.all(hidden.grad[12:14] == .5))
        self.assertEqual(hidden.grad[:12].abs().sum().item(), 0)

    def test_frame_count_matches_three_convolutions_for_partial_chunks(self):
        for mel_length in (1, 7, 8, 9, 99, 100, 101, 110, 199, 200, 201):
            selected = interval_frame_indices([(0, 10)], mel_length, 100, .01)[0]
            expected = 0
            for offset in range(0, mel_length, 100):
                length = min(100, mel_length - offset)
                for _ in range(3):
                    length = (length - 1) // 2 + 1
                expected += length
            self.assertEqual(len(selected), expected)

    def test_mfa_contrastive_branch_uses_boundaries_and_backpropagates(self):
        from scripts.baseline.finetuning_pinyin_ctc_qwen3 import Qwen3CTCModule

        def forbid_ctc(*args):
            raise AssertionError('MFA pooling must not compute CTC posteriors')

        model = SimpleNamespace(
            args=SimpleNamespace(mfa_alignment_dir='mfa', tone_temperature=.1),
            tone_acoustic_only=True,
            phones=SimpleNamespace(id_to_phone={3: 'i1', 4: 'i2'}),
            ctc_occurrence_posteriors=forbid_ctc, log=lambda *args, **kwargs: None)
        hidden = torch.randn(4, 2, 8, requires_grad=True)
        logits = torch.randn(4, 2, 5, requires_grad=True)
        rows = [dict(source='test', speaker_id=str(i), mfa_frame_indices=[[0, 1]]) for i in range(3)]
        # Missing MFA: contributes neither positive nor negative instances.
        rows.append(dict(source='test', speaker_id='3'))
        loss, _, anchors = Qwen3CTCModule.tone_contrastive_loss(
            model, hidden, logits, torch.tensor([2]*4), [[3], [3], [4], [3]], rows)
        self.assertEqual(anchors, 2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(hidden.grad[:3].abs().sum().item(), 0)
        self.assertEqual(hidden.grad[3].abs().sum().item(), 0)
        self.assertIsNone(logits.grad)


if __name__ == '__main__':
    unittest.main()

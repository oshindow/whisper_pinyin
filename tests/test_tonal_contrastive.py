import itertools
import unittest

import torch

from scripts.baseline.tonal_contrastive import occurrence_posteriors, TonalFinalContrastive


class TonalContrastiveTests(unittest.TestCase):
    def test_posterior_matches_exhaustive_paths_with_repeated_labels(self):
        torch.manual_seed(3)
        probs = torch.randn(5, 3).log_softmax(-1)
        for target in ([1, 2], [1, 1]):
            labels = [0]
            for token in target:
                labels.extend([token, 0])
            occupancy = torch.zeros(5, len(target))
            total = 0.0
            for path in itertools.product(range(len(labels)), repeat=5):
                if path[0] not in (0, 1) or path[-1] not in (len(labels)-2, len(labels)-1):
                    continue
                valid = True
                for prev, nxt in zip(path, path[1:]):
                    if nxt-prev not in (0, 1, 2) or (nxt-prev == 2 and (labels[nxt] == 0 or labels[nxt] == labels[prev])):
                        valid = False
                        break
                if not valid:
                    continue
                weight = torch.stack([probs[t, labels[state]] for t, state in enumerate(path)]).sum().exp()
                total += weight
                for t, state in enumerate(path):
                    if state % 2:
                        occupancy[t, state//2] += weight
            torch.testing.assert_close(occurrence_posteriors(probs, target), occupancy / total)
        self.assertIsNone(occurrence_posteriors(probs[:1], [1, 1]))

    def test_gradient_reaches_hidden_and_embeddings_but_not_alignment(self):
        module = TonalFinalContrastive('data/lang_jsonl/tokens.txt', 188, 8)
        hidden = torch.randn(1, 6, 8, requires_grad=True)
        logits = torch.randn(1, 6, 188, requires_grad=True)
        loss, _, count = module(hidden, logits.log_softmax(-1), [[3, 3]])
        self.assertEqual(count, 2)
        loss.backward()
        self.assertGreater(hidden.grad.abs().sum().item(), 0)
        self.assertGreater(module.phone_embedding.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(logits.grad)
        self.assertEqual(module.candidates[14], (14, 15, 16, 17))
        self.assertNotIn(26, module.candidates)  # b: consonant
        used = {i for ids in module.candidates.values() for i in ids}
        self.assertNotIn(26, used)
        self.assertNotIn(0, used)

    def test_empty_tonal_batch_keeps_gradient_graph(self):
        module = TonalFinalContrastive('data/lang_jsonl/tokens.txt', 188, 8)
        hidden = torch.randn(1, 6, 8, requires_grad=True)
        loss, _, count = module(hidden, torch.randn(1, 6, 188).log_softmax(-1), [[26]])
        self.assertEqual(count, 0)
        loss.backward()
        self.assertIsNotNone(module.phone_embedding.weight.grad)


if __name__ == '__main__':
    unittest.main()

"""Phone-prototype supervision using detached CTC occurrence posteriors."""
import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def occurrence_posteriors(log_probs, targets, blank=0):
    """Return [time, target occurrence] occupancy, or None if unalignable."""
    log_probs = log_probs.detach().float()
    targets = torch.as_tensor(targets, device=log_probs.device, dtype=torch.long)
    time = log_probs.shape[0]
    if time == 0 or targets.numel() == 0:
        return None
    labels = targets.new_full((2 * targets.numel() + 1,), blank)
    labels[1::2] = targets
    emissions = log_probs[:, labels]
    skip = torch.zeros_like(labels, dtype=torch.bool)
    skip[2:] = (labels[2:] != blank) & (labels[2:] != labels[:-2])
    neg = emissions.new_full((1,), -torch.inf)
    alpha = emissions.new_full(emissions.shape, -torch.inf)
    alpha[0, :2] = emissions[0, :2]
    for t in range(1, time):
        prev = alpha[t - 1]
        one = torch.cat((neg, prev[:-1]))
        two = torch.cat((neg.expand(2), prev[:-2])).masked_fill(~skip, -torch.inf)
        alpha[t] = torch.logsumexp(torch.stack((prev, one, two)), 0) + emissions[t]
    normalizer = torch.logsumexp(alpha[-1, -2:], 0)
    if not torch.isfinite(normalizer):
        return None
    beta = emissions.new_full(emissions.shape, -torch.inf)
    beta[-1, -2:] = 0
    for t in range(time - 2, -1, -1):
        nxt = beta[t + 1] + emissions[t + 1]
        one = torch.cat((nxt[1:], neg))
        two = torch.cat((nxt[2:].masked_fill(~skip[2:], -torch.inf), neg.expand(2)))
        beta[t] = torch.logsumexp(torch.stack((nxt, one, two)), 0)
    return (alpha[:, 1::2] + beta[:, 1::2] - normalizer).exp()


class TonalFinalContrastive(nn.Module):
    def __init__(self, token_path, vocab_size, hidden_size, temperature=0.1):
        super().__init__()
        if temperature <= 0:
            raise ValueError("tone temperature must be positive")
        self.temperature = temperature
        self.phone_embedding = nn.Embedding(vocab_size, hidden_size)
        groups = {}
        for line in Path(token_path).read_text(encoding="utf-8").splitlines():
            token, index = line.split()
            match = re.fullmatch(r"(.+)([1-5])", token)
            if match:
                groups.setdefault(match[1], []).append(int(index))
        self.candidates = {
            index: tuple(sorted(indices))
            for indices in groups.values() if len(indices) > 1
            for index in indices
        }

    def forward(self, hidden, log_probs, targets, input_lengths=None):
        # Keep the embedding in the DDP graph even during warmup or empty batches.
        zero = hidden.float().sum() * 0 + self.phone_embedding.weight.sum() * 0
        losses, correct = [], zero.detach()
        for b, sequence in enumerate(targets):
            positions = [u for u, token in enumerate(sequence) if token in self.candidates]
            if not positions:
                continue
            length = hidden.shape[1] if input_lengths is None else int(input_lengths[b])
            posterior = occurrence_posteriors(log_probs[b, :length], sequence)
            if posterior is None:
                continue
            weights = posterior[:, positions]
            masses = weights.sum(0)
            pooled = weights.T @ hidden[b, :length].float()
            pooled = F.normalize(pooled / masses.clamp_min(1e-8).unsqueeze(1), dim=-1)
            for i, u in enumerate(positions):
                if masses[i] <= 1e-8:
                    continue
                ids = self.candidates[sequence[u]]
                prototypes = F.normalize(self.phone_embedding.weight[list(ids)].float(), dim=-1)
                scores = pooled[i] @ prototypes.T / self.temperature
                label = ids.index(sequence[u])
                losses.append(F.cross_entropy(scores.unsqueeze(0), scores.new_tensor([label], dtype=torch.long)))
                correct = correct + (scores.argmax() == label).float()
        if not losses:
            return zero, zero.detach(), 0
        return torch.stack(losses).mean() + zero, correct / len(losses), len(losses)

"""Voiced-frame supervision for normalized log-F0 (not Hz regression)."""
import torch
import torch.nn.functional as F


def f0_loss_stats(prediction, features):
    """Return Smooth L1 sum, absolute error sum and voiced-frame count.

    Channel 0 is the normalized contour; channel 2 is the voiced mask.
    Cache padding and interpolated unvoiced frames have mask zero. Empty
    voiced batches return differentiable zero, keeping the head in DDP.
    """
    target = features[:, 0].detach().float()
    if prediction.shape != target.shape:
        raise ValueError('F0 prediction and target frame shapes must match')
    mask = features[:, 2] > 0.5
    predicted = prediction.float()[mask]
    target = target[mask]
    return (F.smooth_l1_loss(predicted, target, reduction='sum'),
            (predicted - target).abs().sum(), mask.sum())

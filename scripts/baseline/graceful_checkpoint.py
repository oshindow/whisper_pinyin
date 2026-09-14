"""Coordinate signal-triggered checkpoints at optimizer update boundaries."""
import signal

import torch
from pytorch_lightning.callbacks import Callback


class GracefulCheckpoint(Callback):
    def __init__(self, path):
        self.path = str(path)
        self.requested = False
        self.pending = False
        self.previous_step = 0
        self.handlers = {}

    def _request_stop(self, signum, frame):
        # Signal handlers must never run distributed collectives or disk writes.
        self.requested = True

    def on_train_start(self, trainer, pl_module):
        self.previous_step = trainer.global_step
        # Install after Lightning's handlers; coordinate ranks in batch hooks.
        for signum in (signal.SIGINT, signal.SIGTERM):
            self.handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._request_stop)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        requested = torch.tensor(int(self.requested), device=pl_module.device)
        requested = trainer.strategy.reduce(requested, reduce_op="sum")
        self.pending = self.pending or bool(requested.item())
        updated = trainer.global_step > self.previous_step
        self.previous_step = trainer.global_step
        if self.pending and updated:
            # Stop only after accumulated gradients have been applied. All ranks
            # participate in save_checkpoint at train end, after loop bookkeeping.
            trainer.should_stop = True

    def on_train_end(self, trainer, pl_module):
        trainer.save_checkpoint(self.path, weights_only=False)
        if trainer.is_global_zero:
            print(f"Saved full training checkpoint at step {trainer.global_step}: {self.path}", flush=True)

    def teardown(self, trainer, pl_module, stage):
        for signum, handler in self.handlers.items():
            signal.signal(signum, handler)
        self.handlers.clear()

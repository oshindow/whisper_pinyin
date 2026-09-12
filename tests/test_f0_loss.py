import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from pytorch_lightning import LightningModule
import whisper
from whisper.model import CTCHead, ModelDimensions, Whisper
from preprocessing.f0_features import f0_to_features
from preprocessing.f0_loss import f0_loss_stats
from scripts.baseline.finetuning_pinyin_ctc_torch import WhisperModelModule


def model():
    result = Whisper(ModelDimensions(80, 8, 64, 1, 1, 16, 8, 64, 1, 1),
                     ctc_vocab=8, use_f0=True, f0_dim=16)
    for name in ('decoder', 'stct_head', 'accent_classifier', 'acc_head',
                 'accent_embedding', 'decoder_projection'):
        setattr(result, name, None)
    return result


class F0LossTests(unittest.TestCase):
    def test_mask_and_empty(self):
        features = torch.tensor(f0_to_features([100, 0, 200], 8))[None]
        prediction = features[:, 0].clone().requires_grad_()
        error, mae, count = f0_loss_stats(prediction, features)
        self.assertEqual(count.item(), 2)
        self.assertEqual(error.item(), 0)
        modified = prediction.detach().clone()
        modified[:, 1] = 100
        modified[:, 3:] = 100
        self.assertEqual(f0_loss_stats(modified, features)[0].item(), 0)
        empty = torch.zeros_like(features)
        loss = f0_loss_stats(prediction, empty)[0]
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def module(self):
        result = WhisperModelModule.__new__(WhisperModelModule)
        LightningModule.__init__(result)
        result.model = model()
        result.model.f0_head = CTCHead(64, 1, 2)
        result.f0_loss_weight = 0.1
        result.encoder_dropout = torch.nn.Identity()
        result.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)
        result.lexicon = {'n': ['n'], 'i3': ['i3']}
        result.token_table_tight = {'<sos/eos>': 1, 'n': 3, 'i3': 4}
        result.cfg = SimpleNamespace(learning_rate=5e-5, adam_epsilon=1e-8,
                                    head_warmup_steps=3000, freeze_backbone_steps=10000,
                                    backbone_ramp_steps=3000, max_steps=30000)
        return result

    def batch(self):
        return {'input_ids': torch.randn(2, 80, 16), 'pinyins': ['n i3', 'n i3'],
                'f0': torch.tensor(f0_to_features([100, 120, 0, 180], 8))[None].repeat(2, 1, 1)}

    def test_no_f0_input_shortcut(self):
        module = self.module().eval()
        batch = self.batch()
        _, prediction = module.encode(batch, return_f0_prediction=True)
        changed = dict(batch, f0=batch['f0'] + 10)
        _, other = module.encode(changed, return_f0_prediction=True)
        torch.testing.assert_close(prediction, other)
        f0_loss_stats(prediction, batch['f0'])[0].backward()
        self.assertGreater(module.model.encoder.conv1.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in module.model.f0_encoder.parameters()))

    def test_joint_loss_and_optimizer(self):
        module = self.module()
        logs = {}
        module.log = lambda name, value, **kw: logs.update({name: value})
        loss = module.training_step(self.batch(), 0)
        torch.testing.assert_close(loss, logs['train/ctc_loss'] + 0.1 * logs['train/f0_loss'])
        loss.backward()
        for head in (module.model.f0_head, module.model.ctc_head, module.model.f0_encoder):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()))
        optimizers, _ = module.configure_optimizers()
        params = {id(p) for group in optimizers[0].param_groups for p in group['params']}
        self.assertTrue(all(id(p) in params for p in module.model.f0_head.parameters()))
        module.f0_loss_weight = 0
        logs.clear()
        loss = module.training_step(self.batch(), 0)
        torch.testing.assert_close(loss, logs['train/ctc_loss'])
        self.assertNotIn('train/f0_loss', logs)

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            for auxiliary in (False, True):
                original = model().eval()
                if auxiliary:
                    original.f0_head = CTCHead(64, 1, 2)
                path = Path(directory) / 'model.ckpt'
                torch.save({'state_dict': {'model.' + k: v for k, v in original.state_dict().items()}}, path)
                loaded = whisper.load_model(str(path), device='cpu', ctc_vocab=8,
                                            use_f0=True, f0_dim=16).eval()
                self.assertEqual(hasattr(loaded, 'f0_head'), auxiliary)
                for key, value in original.state_dict().items():
                    torch.testing.assert_close(value, loaded.state_dict()[key])

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.distributed as dist
import whisper
 
from pathlib import Path
from preprocessing.phone_units import finals_from_lexicon, merge_units
from preprocessing.preprocess_pinyin import WhisperPinyinDataset, WhisperDataCollatorWhithPadding
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import WhisperTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import Config
 
import k2

import random
import numpy as np
import argparse

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

SEED = 2025
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seed(SEED)

class WhisperModelModule(LightningModule):
    def __init__(self, cfg:Config, model_name="large", lang="zh") -> None:
        super().__init__()

        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3-turbo", language=lang, task="transcribe")
        self.model = self.load_pretrained_whisper(model_name, ctc_vocab=cfg.vocab_size, ctc_layers=cfg.ctc_layers)

        self.train_path = cfg.train_path
        self.val_path = cfg.val_path
        self.test_path = cfg.test_path
        self.spk_info_path = cfg.spk_info_path
        
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

        self.cfg = cfg
        self.train_dataset = WhisperPinyinDataset(self.train_path, self.tokenizer, self.spk_info_path, config=self.cfg, task='train',
                                             pseudo_labels=getattr(self, 'pseudo_labels', None))
        
        
        self.lexicon = self.get_lexicon(lexicon_path=cfg.lexicon_path)
        # Empty unless the lexicon splits tonal finals into a final plus a tone.
        self.unit_finals = finals_from_lexicon(self.lexicon)
        self.token_table_tight = k2.SymbolTable.from_file(cfg.token_table_path)
        print(cfg.token_table_path)

        # This recipe is encoder-only. Remove Whisper's unused decoder and
        # auxiliary heads before DDP wraps the module.
        for name in ("decoder", "stct_head", "accent_classifier", "acc_head",
                     "accent_embedding", "decoder_projection"):
            if hasattr(self.model, name):
                setattr(self.model, name, None)

        # The convolutional feature frontend is permanently frozen. Transformer
        # blocks remain in DDP from the start and are frozen by LR=0 for 10k steps.
        for parameter in list(self.model.encoder.conv1.parameters()) + list(self.model.encoder.conv2.parameters()):
            parameter.requires_grad = False

        for block in self.model.encoder.blocks:
            block.attn_dropout = nn.Dropout(cfg.dropout)
            block.mlp_dropout = nn.Dropout(cfg.dropout)
        self.encoder_dropout = nn.Dropout(cfg.dropout)
        self.model.ctc_head.final_dropout = nn.Dropout(cfg.dropout)

        self.register_buffer("val_errors", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("val_ref_phones", torch.tensor(0, dtype=torch.long), persistent=False)
        

    def get_lexicon(self, lexicon_path):
        lexicon = {}
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split(' ')
                word = line[0]
                tokens = line[1:]
                lexicon[word] = tokens
        return lexicon 
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        
        input_ids = batch["input_ids"]
        pinyins = batch["pinyins"] 
        
        audio_features,_ = self.model.encoder(input_ids)
        audio_features = self.encoder_dropout(audio_features)
            
        # ctc head
        _, nnet_output = self.model.ctc_head(audio_features)

        nnet_output_log = nnet_output.log_softmax(2)
        log_probs = nnet_output_log.transpose(0, 1) # L, B, D
        
        ctc_labels = []
        for sentence in pinyins:
            # torch.nn.CTCLoss does not add boundary symbols. Keep exactly one
            # <sos/eos> at each end; k2 graph compilation is not used here.
            sentence_labels = [self.token_table_tight['<sos/eos>']]
            for word in sentence.split(' '):
                if word not in self.lexicon:
                    print(f"Warning: {word} not in lexicon")
                    continue
                for piece in self.lexicon[word]:
                    if piece not in self.token_table_tight:
                        continue
                    sentence_labels.append(self.token_table_tight[piece])
            sentence_labels.append(self.token_table_tight['<sos/eos>'])
            ctc_labels.append(sentence_labels)
        
        targets = torch.tensor(
            [item for sublist in ctc_labels for item in sublist],
            dtype=torch.long
        ).to(self.device)

        target_lengths = torch.tensor(
            [len(x) for x in ctc_labels],
            dtype=torch.long
        ).to(self.device)

        output_lengths = torch.tensor(
            [nnet_output.shape[1]] * nnet_output.shape[0],
            dtype=torch.long
        ).to(self.device)

        

        ctc_loss = self.ctc_loss(
            log_probs,
            targets,
            output_lengths,
            target_lengths
        )
        
         
         
        loss = ctc_loss 
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("train/ctc_loss", ctc_loss, on_step=True, prog_bar=True, logger=True)
                
        return loss

    def validation_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        pinyins = batch["pinyins"] 
        with torch.no_grad():
            audio_features,_ = self.model.encoder(input_ids)
            _, nnet_output = self.model.ctc_head(audio_features)

        texts = self.ctc_decode(nnet_output, pinyins)
        
        errors = 0
        ref_phones = 0
        for o, l in zip(texts, pinyins):
            # PER stays at the phone level whatever the CTC units are: the
            # hypothesis units are merged back into tonal finals and scored
            # against the phones the manifest annotates, so checkpoint
            # selection is comparable across unit inventories.
            hyp = merge_units(o, self.unit_finals)
            ref = [word for word in l.split(' ') if word in self.lexicon]

            errors += self.edit_distance(ref, hyp)
            ref_phones += len(ref)
        self.val_errors += errors
        self.val_ref_phones += ref_phones

    def on_validation_epoch_start(self):
        self.val_errors.zero_()
        self.val_ref_phones.zero_()

    def on_validation_epoch_end(self):
        totals = torch.stack((self.val_errors, self.val_ref_phones))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        per = 100.0 * totals[0].float() / totals[1].clamp_min(1)
        self.log("val/per", per, prog_bar=True, logger=True, sync_dist=False)

    @staticmethod
    def edit_distance(ref, hyp):
        previous = list(range(len(hyp) + 1))
        for i, ref_phone in enumerate(ref, 1):
            current = [i]
            for j, hyp_phone in enumerate(hyp, 1):
                current.append(min(current[-1] + 1, previous[j] + 1,
                                   previous[j - 1] + (ref_phone != hyp_phone)))
            previous = current
        return previous[-1]
    
    def ctc_decode(self, logits, pinyins):
        class_indices = logits.argmax(dim=2)
        texts = []
        idx = 0
        for seq in class_indices:
            seq_collapsed = []
            prev_token = -1
            # Standard CTC: collapse repeats first, then remove blank/specials.
            for token in seq:
                if token != prev_token and token.item() not in (0, 1, 2):
                    seq_collapsed.append(token.item())
                prev_token = token
            
            # Decode to text
            text = [self.token_table_tight._id2sym[i] for i in seq_collapsed]
            texts.append(text)
            idx += 1
        return texts
    
    def configure_optimizers(self):
        backbone = [p for name, p in self.model.encoder.named_parameters()
                    if p.requires_grad and not name.startswith(("conv1.", "conv2."))]
        head = list(self.model.ctc_head.parameters())
        optimizer = AdamW(
            [{"params": backbone, "name": "backbone"},
             {"params": head, "name": "ctc_head"}],
            lr=self.cfg.learning_rate, betas=(0.9, 0.98),
            eps=self.cfg.adam_epsilon, weight_decay=0.0,
        )
        self.optimizer = optimizer

        def head_lr(step):
            if step < self.cfg.head_warmup_steps:
                return step / max(1, self.cfg.head_warmup_steps)
            return max(0.0, (self.cfg.max_steps - step) /
                       max(1, self.cfg.max_steps - self.cfg.head_warmup_steps))

        def backbone_lr(step):
            if step < self.cfg.freeze_backbone_steps:
                return 0.0
            ramp_end = self.cfg.freeze_backbone_steps + self.cfg.backbone_ramp_steps
            if step < ramp_end:
                return (step - self.cfg.freeze_backbone_steps) / max(1, self.cfg.backbone_ramp_steps)
            return max(0.0, (self.cfg.max_steps - step) /
                       max(1, self.cfg.max_steps - ramp_end))

        scheduler = LambdaLR(optimizer, lr_lambda=[backbone_lr, head_lr])
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            print('total train dataset length:', len(self.train_dataset.datalist))

    def load_pretrained_whisper(self, model_name, ctc_vocab, ctc_layers):
        # Load the original Whisper model
        model = whisper.load_model(model_name, ctc_vocab=ctc_vocab, ctc_layers=ctc_layers)

        print("Loaded Whisper model and initialized missing weights.")

        return model  # Return the modified model
        
    def train_dataloader(self):
        
        return torch.utils.data.DataLoader(self.train_dataset, 
                          batch_size=self.cfg.batch_size, 
                          drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )

    def val_dataloader(self):
        dataset = WhisperPinyinDataset(self.val_path, self.tokenizer, self.spk_info_path, config=self.cfg, task='dev')
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.cfg.batch_size, 
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )


if __name__ == '__main__':
    
    print(whisper.available_models())
   

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--train-name",
        type=str,
        default="whisper_small_ctc_k2_otc",
        help="Supervision manifest that contains verbatim transcript",
    )
    parser.add_argument(
        "--train-id",
        type=str,
        default="001",
        help="Supervision manifest that contains verbatim transcript",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=10,
        help="Supervision manifest that contains verbatim transcript",
    )

    parser.add_argument(
        "--initial-bypass-weight",
        type=int,
        default=-19,
        help="",
    )
    parser.add_argument(
        "--initial-self-loop-weight",
        type=float,
        default=3.75,
        help="",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="Batch size for training",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="train_data_clean_sub0.04_ins0.03_del0.03",
        help="Batch size for training",
    )
    parser.add_argument("--val-path", type=str, default=Config.val_path)
    parser.add_argument("--lexicon-path", type=str, default=Config.lexicon_path)
    parser.add_argument("--token-table-path", type=str, default=Config.token_table_path)
    parser.add_argument("--ctc-vocab", type=int, default=Config.vocab_size)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--freeze-backbone-steps", type=int, default=10000)
    parser.add_argument("--head-warmup-steps", type=int, default=3000)
    parser.add_argument("--backbone-ramp-steps", type=int, default=3000)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory used to resolve AISHELL-3 audio paths when manifests contain utterance IDs.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="small",
        help="Model name to use for training",
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        default=80,
        help="Number of mel frequency bins",
    )
    parser.add_argument(
        "--ctc-layers",
        type=int,
        default=2,
        help="",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        help="Precision for training, e.g., '16', '32', 'bf16-mixed', 'fp16', 'bf16'",   
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Weight decay for the optimizer",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--adam-epsilon",
        type=float,
        default=1e-6,
        help="Epsilon for the Adam optimizer",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
        help="Number of warmup steps for the learning rate scheduler",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="exp2",
        help="Root directory for checkpoints and TensorBoard logs",
    )

    args = parser.parse_args()
    train_name = args.train_name
    train_id = args.train_id
    model_name = args.model_name

    lang = "zh"
    
    log_output_dir = Path(args.exp_dir) / train_name
    check_output_dir = Path(args.exp_dir) / train_name / train_id
    print("train-path:", args.train_path)
    print("check_output_dir:", check_output_dir)

    cfg = Config()
    assert "▁" not in cfg.otc_token
    cfg.otc_token = f"▁{cfg.otc_token}"

    cfg.num_train_epochs = args.epoch
    cfg.initial_bypass_weight = args.initial_bypass_weight
    cfg.initial_self_loop_weight = args.initial_self_loop_weight
    cfg.batch_size = args.batch_size
    cfg.train_path = args.train_path
    cfg.val_path = args.val_path
    cfg.data_root = args.data_root
    cfg.n_mels = args.n_mels
    cfg.ctc_layers = args.ctc_layers
    cfg.lexicon_path = args.lexicon_path
    cfg.token_table_path = args.token_table_path
    cfg.vocab_size = args.ctc_vocab
    cfg.num_devices = args.devices
    cfg.gradient_accumulation_steps = args.gradient_accumulation_steps
    cfg.max_steps = args.max_steps
    cfg.freeze_backbone_steps = args.freeze_backbone_steps
    cfg.head_warmup_steps = args.head_warmup_steps
    cfg.backbone_ramp_steps = args.backbone_ramp_steps
    cfg.eval_steps = args.eval_steps
    cfg.dropout = args.dropout
    cfg.learning_rate = args.learning_rate
    cfg.weight_decay = args.weight_decay
    cfg.adam_epsilon = args.adam_epsilon
    cfg.warmup_steps = args.warmup_steps

    

    Path(log_output_dir).mkdir(parents=True, exist_ok=True)
    Path(check_output_dir).mkdir(parents=True, exist_ok=True)

    tflogger = TensorBoardLogger(
        save_dir=log_output_dir,
        name="logs",
        version=train_id
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{check_output_dir}",
        filename="checkpoint-step={step:06d}-per={val/per:.2f}",
        monitor="val/per",
        mode="min",
        save_top_k=2,
        save_last=True,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )

    seed_everything(args.seed, workers=True)
    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="step")]
    model = WhisperModelModule(cfg, model_name, lang)

    DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(
        precision=args.precision,
        accelerator=DEVICE,
        devices=args.devices if DEVICE == "gpu" else 1,
        strategy=args.strategy if DEVICE == "gpu" else "auto",
        max_steps=cfg.max_steps,
        max_epochs=-1,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        logger=tflogger,
        callbacks=callback_list,
        # One virtual epoch is exactly 1000 optimizer steps, so validation and
        # checkpointing occur on the requested optimizer-step boundary.
        limit_train_batches=cfg.eval_steps * cfg.gradient_accumulation_steps,
        val_check_interval=1.0,
        check_val_every_n_epoch=1,
    )

    trainer.fit(model)

 
import torch
import torch.nn as nn
import whisper
 
from pathlib import Path
from utils import error_stats 
from preprocess_pinyin import WhisperPinyinDataset, WhisperDataCollatorWhithPadding
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import WhisperTokenizer
from transformers import (
    AdamW,
    get_cosine_schedule_with_warmup
)

from config import Config
 
import k2

from otc_graph_compiler import OtcTrainingGraphCompiler
 
 
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
        self.model = self.load_pretrained_whisper(model_name, ctc_vocab=280, ctc_layers=cfg.ctc_layers)

        self.train_path = cfg.train_path
        self.val_path = cfg.val_path
        self.test_path = cfg.test_path
        self.spk_info_path = cfg.spk_info_path
 
        self.cfg = cfg
        
        self.train_dataset = WhisperPinyinDataset(self.train_path, self.tokenizer, self.spk_info_path, config=self.cfg, task='train',
                                             pseudo_labels=getattr(self, 'pseudo_labels', None))
        
        
        self.lexicon = self.get_lexicon(lexicon_path='data/lang_whisper/lexicon_add_plus_new.txt')
        self.token_table_tight = k2.SymbolTable.from_file(Path('data/whisper') / "tokens.txt")
        print(Path('data/whisper') / "tokens.txt")

        
        self.graph_compiler = OtcTrainingGraphCompiler(
            'data/whisper',
            otc_token=self.cfg.otc_token,
            device=self.device,
            initial_bypass_weight=self.cfg.initial_bypass_weight,
            initial_self_loop_weight=self.cfg.initial_self_loop_weight,
            bypass_weight_decay=self.cfg.bypass_weight_decay,
            self_loop_weight_decay=self.cfg.self_loop_weight_decay,
        )

        

        self.decoding_graph = None
        

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
            
        # ctc head
        _, nnet_output = self.model.ctc_head(audio_features)

        nnet_output_log = nnet_output.log_softmax(2)
        
        # otc token prob
        batch_size, _, V = nnet_output.shape
        
        otc_token_log_prob = torch.logsumexp(
            nnet_output_log[:, :, 1:], dim=-1, keepdim=True
        ) - torch.log(torch.tensor([V - 1])).to(self.device)

        nnet_output_cat = torch.cat([nnet_output_log, otc_token_log_prob], dim=-1)
         
        supervision_segments = torch.stack([
            torch.arange(batch_size, dtype=torch.int32),
            torch.zeros(batch_size, dtype=torch.int32),
            torch.full((batch_size,), 1500, dtype=torch.int32)
        ], dim=1).to(self.device)
        
        bypass_weight = self.graph_compiler.initial_bypass_weight * (
            self.graph_compiler.bypass_weight_decay ** (self.current_epoch - 1)
        )
        self_loop_weight = self.graph_compiler.initial_self_loop_weight * (
            self.graph_compiler.self_loop_weight_decay ** (self.current_epoch - 1)
        )
         
        otc_labels = []
        target_lengths = []
 
        for sentence in pinyins:
            sentence_labels = [[self.token_table_tight['<sos/eos>']]]
            length = 2 # 0
            for word in sentence.split(' '):
                 
                if word not in self.lexicon:
                    continue
                word_labels = []
                for piece in self.lexicon[word]:
                    if piece not in self.token_table_tight:
                        continue
                    word_labels.append(self.token_table_tight[piece])
                
                length += len(word_labels)
                sentence_labels.append(word_labels)
            sentence_labels.append([self.token_table_tight['<sos/eos>']])
            target_lengths.append(length)
            otc_labels.append(sentence_labels)
       
        self.decoding_graph = self.graph_compiler.compile(
            texts=pinyins,
            word_ids_list=otc_labels,
            allow_bypass_arc=self.cfg.allow_bypass_arc,
            allow_self_loop_arc=self.cfg.allow_self_loop_arc,
            bypass_weight=bypass_weight,
            self_loop_weight=self_loop_weight,
        )
         
        target_lengths = torch.tensor(
            target_lengths, dtype=torch.int32, device=self.device
        )
        
        supervision_segments = supervision_segments.to('cpu')
        self.decoding_graph = self.decoding_graph.to(self.device)
        self.dense_fsa_vec = k2.DenseFsaVec(
            nnet_output_cat,
            supervision_segments,
            allow_truncate=3,
        )

        otc_loss = k2.ctc_loss(
            decoding_graph=self.decoding_graph,
            dense_fsa_vec=self.dense_fsa_vec,
            target_lengths=target_lengths,
            output_beam=self.cfg.beam_size,
            reduction=self.cfg.reduction,
            use_double_scores=self.cfg.use_double_scores,
        )

        loss = otc_loss 
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("train/otc_loss", otc_loss, on_step=True, prog_bar=True, logger=True)
               
        return loss

    def validation_step(self, batch, batch_id):
        uids = batch["uids"]
        input_ids = batch["input_ids"]
        pinyins = batch["pinyins"] 
        with torch.no_grad():
            audio_features,_ = self.model.encoder(input_ids)
            _, nnet_output = self.model.ctc_head(audio_features)

        texts = self.ctc_decode(nnet_output, pinyins)
        
        cer_results = []
        for u, o, l in zip(uids, texts, pinyins):
            o_list = o
            l_list = []
            for word in l.split(' '):
                if word not in self.lexicon:
                    continue
                
                for piece in self.lexicon[word]:
                    l_list.append(piece)
             
            cer_results.append((u, l_list, o_list))

        cer = error_stats(f=None, test_set_name="val", results=cer_results, compute_CER=False, sclite_mode=False, enable_log=False)
        
        self.log("val/cer", cer, on_step=True, prog_bar=True, logger=True)
   
     
        return {
            "cer": cer,
        }
    
    def ctc_decode(self, logits, pinyins):
        class_indices = logits.argmax(dim=2)
        texts = []
        idx = 0
        for seq in class_indices:
            # Remove blanks (pad tokens)
            seq_no_blank = seq[seq != 0]
            # Collapse repeats
            seq_collapsed = []
            prev_token = -1
            for token in seq_no_blank:
                if token != prev_token:
                    seq_collapsed.append(token.item())
                    prev_token = token
            
            # Decode to text
            text = [self.token_table_tight._id2sym[i] for i in seq_collapsed if i != 1]
            texts.append(text)
            idx += 1
        return texts
    
    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=self.cfg.learning_rate, 
                          eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer

        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.cfg.warmup_steps, 
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    
    def setup(self, stage=None):
         
        if stage == 'fit' or stage is None:
            print('totol train dataset length:', len(self.train_dataset.datalist))
            self.t_total = (
                (len(self.train_dataset.datalist) // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )

    def load_pretrained_whisper(self, model_name, ctc_vocab, ctc_layers):
        # Load the original Whisper model
        model = whisper.load_model(model_name, ctc_vocab=ctc_vocab, ctc_layers=ctc_layers)
         
        print("Loaded Whisper model and initialized missing weights.")

        return model  
        
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

    args = parser.parse_args()
    train_name = args.train_name
    train_id = args.train_id
    model_name = args.model_name
 
    lang = "zh"
    
    log_output_dir = "exp2/" + train_name
    check_output_dir = "exp2/" + train_name + '/' + train_id
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
    cfg.n_mels = args.n_mels
    cfg.ctc_layers = args.ctc_layers
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
        filename="checkpoint-{epoch:04d}",
        save_top_k=-1 # all model save
    )

    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
    model = WhisperModelModule(cfg, model_name, lang)

    DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

    seed_everything(2025, workers=True)
    trainer = Trainer(
        precision=args.precision,
        accelerator=DEVICE,
        max_epochs=cfg.num_train_epochs,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        logger=tflogger,
        callbacks=callback_list,
        val_check_interval=0.3
    )

    trainer.fit(model)



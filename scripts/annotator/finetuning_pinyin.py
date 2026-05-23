import torch
from preprocessing.preprocess_pinyin import WhisperPinyinDataset, WhisperDataCollatorWhithPadding
from transformers import WhisperTokenizer
import whisper
from utils import error_stats, normlizer
import time
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn as nn
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
from pathlib import Path
import os
SAMPLE_RATE = 16000

class Config:
    learning_rate = 1e-5
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_steps = 50
    batch_size = 16
    num_worker = 2
    num_train_epochs = 10
    gradient_accumulation_steps = 1
    sample_rate = SAMPLE_RATE
    token = 'text'
    data_root = 'data'
    train_path = 'dump/aishell3/train/' + token
    val_path = 'dump/aishell3/val/' + token
    test_path = 'dump/aishell3/test/' + token
    

class WhisperModelModule(LightningModule):
    def __init__(self, cfg:Config, model_name="large", lang="zh") -> None:
        super().__init__()

        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3-turbo", language=lang, task="transcribe")
        initial_prompt = "以下是普通话的句子。"
        initial_prompt_tokens = self.tokenizer.encode(" " + initial_prompt.strip())
        self.options = whisper.DecodingOptions(language="zh", without_timestamps=True, prompt=initial_prompt)
        self.model = whisper.load_model(model_name)

        self.train_path = cfg.train_path
        self.val_path = cfg.val_path
        self.test_path = cfg.test_path
       
        # only decoder training
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        # self.metrics_wer = evaluate.load("wer")
        # self.metrics_cer = evaluate.load("cer")

        self.cfg = cfg
        self.train_dataset_length = 63262

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)

        # logits = self.model.logits(dec_input_ids, audio_features)
        # loss = self.loss_fn(logits.transpose(1,2), labels)

        # method 1
        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_id):
        uids = batch["uids"]
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()


        # audio_features = self.model.encoder(input_ids)
        # TODO: options, 
        out = self.model.decode(input_ids, self.options)
        # out = results.text
        # out = self.model.decoder(dec_input_ids, audio_features) 

        # loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        # out[out == -100] = self.tokenizer.eos_token_id
        # labels[labels == -100] = self.tokenizer.eos_token_id

        # o_list, l_list = [], []
        cer_results = []
        for u, o, l in zip(uids, out, labels):
            # print(self.tokenizer.decode(l, skip_special_tokens=True))
            o_list = o.text.split(' ')
            l_list = self.tokenizer.decode(l, skip_special_tokens=True).split(' ')
            # print(l_list)
            cer_results.append((u, l_list, o_list))

        cer = error_stats(test_set_name="val", results=cer_results, compute_CER=False, sclite_mode=False, enable_log=False)
        print(cer_results, cer)
        # self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val/cer", cer, on_step=True, prog_bar=True, logger=True)
        
        return {
            "cer": cer,
            # "loss": loss
        }

    def configure_optimizers(self):
        """オプティマイザーとスケジューラーを作成する"""
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

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.cfg.warmup_steps, 
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    
    def setup(self, stage=None):
        """初期設定（データセットの読み込み）"""

        if stage == 'fit' or stage is None:
            self.t_total = (
                (self.train_dataset_length // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )
    
    def train_dataloader(self):
        """訓練データローダーを作成する"""
        train_dataset = WhisperPinyinDataset(self.train_path, self.tokenizer, config=self.cfg, task='train')
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=WhisperDataCollatorWhithPadding())

        return torch.utils.data.DataLoader(train_dataset, 
                          batch_size=self.cfg.batch_size, 
                          drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )

    def val_dataloader(self):
        """バリデーションデータローダーを作成する"""
        dataset = WhisperPinyinDataset(self.val_path, self.tokenizer, config=self.cfg, task='test')
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.cfg.batch_size, 
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )


if __name__ == '__main__':
    
    print(whisper.available_models())
    train_name = "whisper"
    train_id = "00002"
    model_name = "turbo"
    lang = "zh"

    log_output_dir = "exp" 
    check_output_dir = "exp" 

    cfg = Config()

    Path(log_output_dir).mkdir(parents=True, exist_ok=True)
    Path(check_output_dir).mkdir(parents=True, exist_ok=True)

    tflogger = TensorBoardLogger(
        save_dir=log_output_dir,
        name=train_name,
        version=train_id
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{check_output_dir}/checkpoint",
        filename="checkpoint-{epoch:04d}",
        save_top_k=-1 # all model save
    )

    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
    model = WhisperModelModule(cfg, model_name, lang)

    DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(
        precision=16,
        accelerator=DEVICE,
        max_epochs=cfg.num_train_epochs,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        logger=tflogger,
        callbacks=callback_list,
        val_check_interval=0.015
    )

    trainer.fit(model)

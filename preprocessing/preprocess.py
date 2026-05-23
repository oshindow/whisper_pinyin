

from pathlib import Path


def get_data_lists(text_path, task='train', data_root='data'):

    samples = []
    with open(text_path, 'r', encoding='utf8') as input:
        for line in input:
            uid = line.strip().split(' ')[0]
            data = line.strip().split(' ')[1:]
            
            texts = [item for item in data]
        
            spk = uid[:7]

            filepath = Path(data_root) / "aishell3" / task / "wav_16k" / spk / f"{uid}.wav"
            filepath = str(filepath)
            samples.append("|".join([filepath, " ".join(texts)]))

    return samples


import random
import torch
import torchaudio as ta
import whisper
import numpy as np

class WhisperPinyinDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, tokenizer, config=None, task='train', random_seed=1020):

        data_root = getattr(config, "data_root", "data")
        self.datalist = get_data_lists(filelist_path, task=task, data_root=data_root)
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()
        self.config = config

        random.seed(random_seed)
        random.shuffle(self.datalist)

    def __getitem__(self, index):
        """
        return:
            input_ids: Tensor (Dim, T) by default dim=80
            dec_input_ids: list, [50260, 50359, 50363, 9572, 220, 24726, 220, 18681, 220, 14028, 26923, 220, 12579, 22933, 220, 6404, 19021, 220, 15686, 50257, 50257] <|startoftranscript|><|zh|><|transcribe|><|notimestamps|>tokens<|endoftext|>
            labels: list, [50258, 50260, 50359, 50363, 9572, 220, 24726, 220, 18681, 220, 14028, 26923, 220, 12579, 22933, 220, 6404, 19021, 220, 15686, 50257] <|zh|><|transcribe|><|notimestamps|>tokens<|endoftext|><|endoftext|>
        """
        audiofile, texts = self.datalist[index].split("|")
        
        uid = audiofile.split('/')[-1][:-4]
        
        texts = texts.replace(' ', '')
        ids = self.tokenizer.encode(texts)[:-1]
        labels = ids[1:] + [self.tokenizer.eos_token_id]
        
        audio, sr = ta.load(audiofile)
        duration = audio.shape[-1] / 16000
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio, n_mels=128) # torch.Size([128, 3000])

        return {
            "durations": duration,
            "uids": uid,
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": ids
        }

    def __len__(self):
        return len(self.datalist)

class WhisperDataCollatorWhithPadding:
    def __call__(self, features):
        durations, uids, input_ids, labels, dec_input_ids = [], [], [], [], []
        for f in features:
            durations.append(f["durations"])
            uids.append(f["uids"]) 
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])
        
        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257 is eot token id

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids
        batch["uids"] = uids
        batch["durations"] = durations

        return batch

import random
import torch
import torchaudio as ta
import whisper
import numpy as np
from pathlib import Path
import os

def get_data_lists(text_path, task='train', data_root='data'):
    """Read a manifest with one sample per line.

    Preferred format:
        /abs/or/relative/audio.wav|pin1 pin2 pin3

    Fallback AISHELL-style format:
        utt_id pin1 pin2 pin3
    """

    samples = []
    text_path = Path(text_path)
    with text_path.open('r', encoding='utf8') as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            if '|' in line:
                samples.append(line)
                continue

            uid, *tokens = line.split()
            spk = uid[:7]
            audio_path = Path(data_root) / 'aishell3' / task / 'wav_16k' / spk / f'{uid}.wav'
            samples.append('|'.join([str(audio_path), ' '.join(tokens)]))

    return samples

class WhisperPinyinDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filelist_path,
        tokenizer,
        spk_info_path=None,
        config=None,
        task='train',
        pseudo_labels=None,
        random_seed=1020,
    ):

        data_root = getattr(config, 'data_root', 'data')
        self.datalist = get_data_lists(filelist_path, task=task, data_root=data_root)
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()
        self.config = config
        self.pseudo_labels = pseudo_labels

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
        
        # texts = texts.replace(' ', '')
        ids = self.tokenizer.encode(texts)[:-1]
        # texts = self.tokenizer.decode(ids)
        if self.pseudo_labels and uid in self.pseudo_labels:
            labels = self.pseudo_labels[uid]
        else:
            labels = ids[1:] + [self.tokenizer.eos_token_id]
        
        if not os.path.isfile(audiofile):
            
            part = audiofile.split('/')[4]  
                  
            audiofile1 = audiofile.replace(part, 'dev')
            audiofile2 = audiofile.replace(part, 'train')
            audiofile3 = audiofile.replace(part, 'test')
            if os.path.isfile(audiofile1):
                # print("is", audiofile1)
                audiofile = audiofile1
            elif os.path.isfile(audiofile2):
                # print("is", audiofile2)
                audiofile = audiofile2
            elif os.path.isfile(audiofile3):
                # print("is", audiofile3)
                audiofile = audiofile3
             

        audio, sr = ta.load(audiofile)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != 16000:
            audio = ta.transforms.Resample(sr, 16000)(audio)

        n_mels = getattr(self.config, "n_mels", 128)
        duration = audio.shape[-1] / 16000
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)
        mel_lens = min(round(audio.shape[-1] / 160 + 0.5), 1500)
        

        return {
            "durations": duration,
            "mel_lens": mel_lens,
            "uids": uid,
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": ids,
            "pinyins": texts,
        }

    def __len__(self):
        return len(self.datalist)

class WhisperDataCollatorWhithPadding:
    def __call__(self, features):
        durations, uids, input_ids, labels, dec_input_ids, pinyins, mel_lens = [], [], [], [], [], [], []
        for f in features:
            durations.append(f["durations"])
            uids.append(f["uids"]) 
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            mel_lens.append(f["mel_lens"])
            dec_input_ids.append(f["dec_input_ids"])
            pinyins.append(f["pinyins"])

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
        batch["pinyins"] = pinyins
        batch["mel_lens"] = mel_lens

        return batch

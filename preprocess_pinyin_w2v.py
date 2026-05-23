
import random
import torch
import torchaudio as ta
import whisper
import numpy as np
import os
from transformers import Wav2Vec2Model, Wav2Vec2Processor

def get_data_lists(text_paths, task='train'):

    samples = []
    if task == 'train':
        with open(text_paths, 'r', encoding='utf-8-sig') as input:
            for line in input:
                if 'asr_chinese' in line:
                    line = line.replace('train', 'train/')
                samples.append(line.strip())
    return samples

class W2vPinyinDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_paths, tokenizer, spk_info_path, config, task='train', pseudo_labels=None, random_seed=1020):
        print(filelist_paths)
        self.datalist = get_data_lists(filelist_paths, task=task)
        self.task = task
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()

        self.pseudo_labels = pseudo_labels
        self.config = config
        random.seed(random_seed)
        random.shuffle(self.datalist)
        
    def __getitem__(self, index):
        """
        return:
            input_ids: Tensor (Dim, T) by default dim=80
            dec_input_ids: list, [50260, 50359, 50363, 9572, 220, 24726, 220, 18681, 220, 14028, 26923, 220, 12579, 22933, 220, 6404, 19021, 220, 15686] <|startoftranscript|><|zh|><|transcribe|><|notimestamps|>tokens
            labels: list, [50258, 50260, 50359, 50363, 9572, 220, 24726, 220, 18681, 220, 14028, 26923, 220, 12579, 22933, 220, 6404, 19021, 220, 15686, 50257] <|zh|><|transcribe|><|notimestamps|>tokens<|endoftext|>
        """
        
        try:
            audiofile, texts ,_,_  = self.datalist[index].split("|")
        except:
            audiofile, texts  = self.datalist[index].split("|")
        uid = audiofile.split('/')[-1][:-4]
         
        ids = self.tokenizer.encode(texts)[:-1]
        
        if self.pseudo_labels and uid in self.pseudo_labels:
            labels = self.pseudo_labels[uid]
        else:
            labels = ids[1:] + [self.tokenizer.eos_token_id]

          
        if not os.path.isfile(audiofile):
            part = audiofile.split('/')[5]  
           
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

        # resample if not 16kHz
        if sr != 16000:
            resampler = ta.transforms.Resample(sr, 16000)
            audio = resampler(audio)
             
        
        duration = audio.shape[-1] / 16000
        mel_lens = min(round(audio.shape[-1] / 160 + 0.5), 1500)
        
        inputs = audio.squeeze().numpy()  # Tensor (T,)
        return {
            "durations": duration,
            "mel_lens": mel_lens,
            "uids": audiofile,
            "input_ids": inputs,
            "labels": labels,
            "dec_input_ids": ids,
            "pinyins": texts,
            
        }

    def __len__(self):
        return len(self.datalist)

class W2vDataCollatorWhithPadding:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base"
        )
        
    def __call__(self, features):
        durations, uids, input_ids, labels, dec_input_ids, pinyins, mel_lens = [], [], [], [], [], [], [] 
        for f in features:
            durations.append(f["durations"])
            uids.append(f["uids"]) 
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            pinyins.append(f["pinyins"])
            mel_lens.append(f["mel_lens"])
        

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)
        
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257 is eot token id
        
        output = self.processor.feature_extractor(
            input_ids,
            padding=True,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True
        )
        # print(output)
        # print(self.processor.feature_extractor.return_attention_mask)
        # ["input_values"].squeeze(0)  # Tensor (T,)
        # print(output["input_values"].shape, output["input_values"])
        # print(output["attention_mask"].shape, output["attention_mask"])
        input_ids = output["input_values"]
        attentin_mask = output["attention_mask"]
        mel_lens = output["input_values"].shape[0]

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids
        batch["attention_mask"] = attentin_mask
        batch["uids"] = uids
        batch["durations"] = durations
        batch["pinyins"] = pinyins
        batch["mel_lens"] = mel_lens

        return batch

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from preprocessing.preprocess import WhisperPinyinDataset, WhisperDataCollatorWhithPadding
from transformers import WhisperTokenizer
import whisper
from utils import error_stats, normlizer
import time
import os
train_text_path = 'dump/aishell3/train/text_char'
train_wave_path = 'dump/aishell3/train/wav.scp'

test_text_path = 'dump/aishell3/test/text_char'
test_wave_path = 'dump/aishell3/test/wav.scp'

# model = whisper.load_model("exp/checkpoint/checkpoint-epoch=0000-v7.ckpt") # good
model = whisper.load_model("exp/checkpoint/checkpoint-epoch=0005.ckpt") 
# model = whisper.load_model("turbo")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3-turbo", language="zh", task="transcribe")
train_dataset = WhisperPinyinDataset(test_text_path, tokenizer, task='test')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=WhisperDataCollatorWhithPadding())
initial_prompt = "以下是普通话的句子。"
initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
options = whisper.DecodingOptions(language="zh", without_timestamps=True, prompt=initial_prompt)

cer_results = []
rtfs = []
cnt = 0
for b in train_loader:
    hypotheses = []
    references = []

    start = time.time()
    with torch.no_grad():
        results = model.decode(b["input_ids"].cuda(), options)
        # out = model.decoder(b["dec_input_ids"].cuda(), model.encoder(b["input_ids"].cuda()))
        # out = torch.argmax(out, dim=1)
        cnt += 1
        if cnt % 10 == 0:
            print(cnt)
            print(cer_results)
            error_stats(test_set_name="test", results=cer_results, compute_CER=True, sclite_mode=False, enable_log=True)
            print("average RTF:", sum(rtfs) / len(rtfs))
        for idx in range(len(results)):
            hypothese = normlizer(results[idx].text)
            # hypothese = tokenizer.decode(out[idx], skip_special_tokens=True)
            reference = tokenizer.decode(b["labels"][idx], skip_special_tokens=True)

            cer_results.append((b['uids'][idx], [item for item in reference], [item for item in hypothese]))
    end = time.time()
    rtf = (end - start) / b["durations"][idx]
    rtfs.append(rtf)
    
# data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))
# data

#     audio_features = model.encoder(b["input_ids"].cuda())
#     input_ids = b["input_ids"] # mel
#     labels = b["labels"].long()
#     dec_input_ids = b["dec_input_ids"].long()


#     audio_features = model.encoder(input_ids.cuda())
#     # print(dec_input_ids)
#     # print(input_ids.shape, dec_input_ids.shape, audio_features.shape)
#     # print(audio_features.shape)
#     # print()
    
# out = model.decoder(dec_input_ids.cuda(), audio_features)
# # print(out.shape)
# # print(out.view(-1, out.size(-1)).shape)
# # print(b["labels"].view(-1).shape)

# tokens = torch.argmax(out, dim=2)
# for token in tokens:
#     token[token == -100] = tokenizer.eos_token_id
#     text = tokenizer.decode(token, skip_special_tokens=True)
    # print(text)

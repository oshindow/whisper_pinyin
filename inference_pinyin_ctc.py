import os
import torch

from preprocess_pinyin import WhisperPinyinDataset, WhisperDataCollatorWhithPadding
from transformers import WhisperTokenizer
import whisper

from utils import error_stats
from config import Config
from pathlib import Path

import k2

def ctc_decode(logits, pinyins):
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
        text = [token_table_tight._id2sym[i] for i in seq_collapsed if i != 1]
        texts.append(text)
        idx += 1
    return texts

import torch


def get_lexicon(lexicon_path):
    lexicon = {}
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            tokens = line[1:]
            lexicon[word] = tokens
    return lexicon 


model_path = 'exp2/whisper_pinyin_cross_ratio_random/001/checkpoint-epoch=0009.ckpt' # whisper-OTC
 
test_paths = ["resources/whisAID/magichub_sg/test_unseen_sorted.csv", "resources/whisAID/magichub_sg/test_seen_sorted.csv",
              "resources/whisAID/latic/test_unseen_sorted.csv", "resources/whisAID/latic/test_seen_sorted.csv"]
for test_path in test_paths:
    Config.test_path = test_path
    dataset = test_path.split('/')[-2]
    error_file = open(f'errorfile_w2v_e9_{dataset}_{os.path.basename(test_path).split(".")[0]}', 'w', encoding='utf8')
    model = whisper.load_model(model_path, ctc_vocab=280, ctc_layers=2) # 296
    print(model_path)
    spk_info_path = 'dump/aishell3/spk_info_only.txt'
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3-turbo", language="zh", task="transcribe")
    test_dataset = WhisperPinyinDataset(Config.test_path, tokenizer, spk_info_path, Config, task='train')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, collate_fn=WhisperDataCollatorWhithPadding())
    
    token_table_tight = k2.SymbolTable.from_file(Path('data/whisper') / "tokens.txt")
    lexicon = get_lexicon(lexicon_path='data/lang_whisper/lexicon_add_plus_new.txt')
    cer_results = []
 
    print(len(test_loader))
  

    for b in test_loader:
        hypotheses = []
        references = []

      
        with torch.no_grad():
            audio_features,_ = model.encoder(b["input_ids"].cuda())
            _, nnet_output = model.ctc_head(audio_features)
            texts = ctc_decode(nnet_output, b["pinyins"])
            
            for u, o, l in zip(b['uids'], texts, b["pinyins"]):
                
                o_list = o
                l_list = []
                for word in l.split(' '):
                    if word not in lexicon:
                        continue
                    
                    for piece in lexicon[word]:
                        l_list.append(piece)
            
                cer_results.append((u, l_list, o_list))

 
    error_stats(f=error_file, test_set_name="test", results=cer_results, compute_CER=False, sclite_mode=False, enable_log=True)
    
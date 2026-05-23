import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import argparse
from preprocessing.preprocess_pinyin_w2v import W2vPinyinDataset, W2vDataCollatorWhithPadding
from transformers import WhisperTokenizer
from scripts.baseline.finetuning_pinyin_w2v import CTCHead
from transformers import Wav2Vec2Model 
import whisper
from utils import error_stats
from config import Config
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

def get_lexicon(lexicon_path):
    lexicon = {}
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            tokens = line[1:]
            lexicon[word] = tokens
    return lexicon 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="hf_model/checkpoint-epoch=0009.ckpt")
    parser.add_argument("--test-path", default=Config.test_path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="errorfile_w2v_aishell3_test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    Config.test_path = args.test_path
    error_file = open(args.output, 'w', encoding='utf8')
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    w2v = Wav2Vec2Model.from_pretrained("TencentGameMate/chinese-wav2vec2-large").to(args.device)
    ctc_head = CTCHead(vocab_size=Config.vocab_size, hidden_size=1024, num_layers=Config.ctc_layers).to(args.device)

    w2v.load_state_dict({k.replace('w2v.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('w2v.')}, strict=False)
    ctc_head.load_state_dict({k.replace('ctc_head.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('ctc_head.')}, strict=False)
    

    spk_info_path = Config.spk_info_path
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3-turbo", language="zh", task="transcribe")
    test_dataset = W2vPinyinDataset(Config.test_path, tokenizer, spk_info_path, Config, task='train')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=W2vDataCollatorWhithPadding())
    
    token_table_tight = k2.SymbolTable.from_file(Config.token_table_path)
    lexicon = get_lexicon(lexicon_path=Config.lexicon_path)
    cer_results = []
    print(len(test_loader))
    
    for b in test_loader:
        hypotheses = []
        references = []

        with torch.no_grad():
            audio_features = w2v(b["input_ids"].to(args.device), attention_mask=b["attention_mask"].to(args.device)).last_hidden_state 
            _, nnet_output = ctc_head(audio_features)
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
    

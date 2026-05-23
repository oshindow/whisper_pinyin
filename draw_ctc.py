import numpy as np
import matplotlib.pyplot as plt
import torch
import k2
from pathlib import Path

def get_lexicon(lexicon_path):
    lexicon = {}
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            tokens = line[1:]
            lexicon[word] = tokens
    return lexicon 


token_table_tight = k2.SymbolTable.from_file(
    Path('data/whisper') / "tokens.txt"
)
lexicon = get_lexicon(lexicon_path='data/lang_whisper/lexicon_add_plus_new.txt')
    
tokens = ['jian3', 'zhi2', 'shi4', 'zi4', 'ji2']
tokens_ids = []
for token in tokens:
    if token not in lexicon:
        print(f"Token {token} is in tokens but not in lexicon")
        pass
    else:
        for piece in lexicon[token]:
            tokens_ids.append(token_table_tight[piece])

blank_id = 0

# ==========================================
# load real probs
# ==========================================
import os
for model in os.listdir('.'):
    if model.endswith('.pt') and 'nnet_output' in model:

        probs = torch.load(
            model
        )
        probs = torch.softmax(probs, dim=-1)
        probs = probs.detach().cpu().numpy()
        T = 60
        V = len(tokens_ids) + 1
        probs = probs[30:, :] 

        selected_tokens = []
        selected_tokens_sym = []
        for t in range(T):

            token_id = np.argmax(probs[t])

            # skip blank
            if token_id == blank_id:
                continue

            if token_id == 1:
                continue
            selected_tokens.append((token_id, t))
            selected_tokens_sym.append(token_table_tight._id2sym[token_id])

        print("selected tokens:", selected_tokens)
        print("selected tokens (symbols):", selected_tokens_sym)

        # ==========================================
        # plotting
        # ==========================================

        fig, ax = plt.subplots(figsize=(9, 3))

        # blank
        ax.plot(
            probs[:, blank_id],
            linestyle='--',
            color='gray',
            linewidth=2,
            alpha=0.8
        )

        colors = plt.cm.tab20.colors

        # ploted
        last_token_id = None
        for idx, (token_id, first_frame) in enumerate(selected_tokens):
            
            curve = probs[:, token_id]

            ax.plot(
                curve,
                linewidth=2,
                color=colors[token_id % len(colors)]
            )

            if token_id != last_token_id:
                ax.text(
                    first_frame,
                    0.03,
                    token_table_tight._id2sym[token_id],
                    fontsize=10,
                    ha='center'
                )
            last_token_id = token_id
        # ==========================================
        # settings
        # ==========================================

        ax.set_xlim(0, T)
        ax.set_ylim(0, 1.05)

        ax.set_xlabel("Frames")
        ax.set_ylabel("Token probability")

        plt.tight_layout()
        plt.savefig(f"{model}_ctc_probs.png", dpi=300)
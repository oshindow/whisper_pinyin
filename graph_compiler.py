# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import List, Union

import k2
import torch

# from icefall.lexicon import Lexicon
from transformers import WhisperTokenizer

class CtcTrainingGraphCompiler(object):
    def __init__(
        self,
        lang_dir: str,
        device: torch.device,
        oov: str = "<UNK>",
        need_repeat_flag: bool = False,
        tokenizer: WhisperTokenizer = None,
    ):
        """
        Args:
          lexicon:
            It is built from `data/lang/lexicon.txt`.
          device:
            The device to use for operations compiling transcripts to FSAs.
          oov:
            Out of vocabulary word. When a word in the transcript
            does not exist in the lexicon, it is replaced with `oov`.
          need_repeat_flag:
            If True, will add an attribute named `_is_repeat_token_` to ctc_topo
            indicating whether this token is a repeat token in ctc graph.
            This attribute is needed to implement delay-penalty for phone-based
            ctc loss. See https://github.com/k2-fsa/k2/pull/1086 for more
            details. Note: The above change MUST be included in k2 to open this
            flag.
        """
        lang_dir = Path(lang_dir)
         
        # self.token_table = k2.SymbolTable.from_file(lang_dir / "tokens_add_new.txt")
        self.token_table = k2.SymbolTable.from_file(lang_dir / "tokens.txt")

        max_token_id = self.get_max_token_id()
        # max_token_id = 50000
        ctc_topo = k2.ctc_topo(max_token_id, modified=False)

        self.ctc_topo = ctc_topo.to(device)

        if need_repeat_flag:
            self.ctc_topo._is_repeat_token_ = (
                self.ctc_topo.labels != self.ctc_topo.aux_labels
            )

        self.device = device
        self.tokenizer = tokenizer

    def get_max_token_id(self):
        # max_token_id = 0
        # for symbol in self.token_table.symbols:
        #     if not symbol.startswith("#"):
        #         max_token_id = max(self.token_table[symbol], max_token_id)
        # assert max_token_id > 0

        return len(self.token_table)
    
    def compile(self, texts: List[str] = None, word_ids_list: List[list] = None) -> k2.Fsa:
        """Build decoding graphs by composing ctc_topo with
        given transcripts.

        Args:
          texts:
            A list of strings. Each string contains a sentence for an utterance.
            A sentence consists of spaces separated words. An example `texts`
            looks like:

                ['hello icefall', 'CTC training with k2']

        Returns:
          An FsaVec, the composition result of `self.ctc_topo` and the
          transcript FSA.
        """
        transcript_fsa = self.convert_transcript_to_fsa(texts, word_ids_list)

        # NOTE: k2.compose runs on CUDA only when treat_epsilons_specially
        # is False, so we add epsilon self-loops here
        fsa_with_self_loops = k2.remove_epsilon_and_add_self_loops(transcript_fsa)

        fsa_with_self_loops = k2.arc_sort(fsa_with_self_loops)

        decoding_graph = k2.compose(
            self.ctc_topo, fsa_with_self_loops, treat_epsilons_specially=False
        )

        assert decoding_graph.requires_grad is False

        return decoding_graph

    def texts_to_ids(self, texts: List[str]) -> List[List[int]]:
        """Convert a list of texts to a list-of-list of word IDs.

        Args:
          texts:
            It is a list of strings. Each string consists of space(s)
            separated words. An example containing two strings is given below:

                ['HELLO ICEFALL', 'HELLO k2']
        Returns:
          Return a list-of-list of word IDs.
        """
        word_ids_list = []
        for text in texts:
            word_ids = []
            for word in text.split():
                if word in self.word_table:
                    word_ids.append(self.word_table[word])
                else:
                    word_ids.append(self.oov_id)
            word_ids_list.append(word_ids)
        return word_ids_list
    def make_arc(
        self,
        from_state: int,
        to_state: int,
        symbol: Union[str, int],
        weight: float,
    ):
        return f"{from_state} {to_state} {symbol} {weight}"
    
    def convert_transcript_to_fsa(self, texts: List[str] = None, word_ids_list: List[List] = None) -> k2.Fsa:
        """Convert a list of transcript texts to an FsaVec.

        Args:
          texts:
            A list of strings. Each string contains a sentence for an utterance.
            A sentence consists of spaces separated words. An example `texts`
            looks like:

                ['hello icefall', 'CTC training with k2']

        Returns:
          Return an FsaVec, whose `shape[0]` equals to `len(texts)`.
        """
        if word_ids_list is None:
            word_ids_list = []
            for text in texts:
                word_ids = []
                for word in text.split():
                    if word in self.word_table:
                        word_ids.append(self.word_table[word])
                    else:
                        word_ids.append(self.oov_id)
                word_ids_list.append(word_ids)
                
        # print(word_ids_list)
        transcript_fsa_list = []
        for word_ids in word_ids_list:
            # one sentence
            arcs = []
            start_state = 0
            cur_state = start_state
            next_state = 1

            for piece_id in word_ids:
                # for piece_id in piece_ids:
                arc = self.make_arc(cur_state, next_state, piece_id, 0.0)
                
                arcs.append(arc)

                cur_state = next_state
                next_state += 1
            
            # Deal with final state
            final_state = next_state
            final_arc = self.make_arc(cur_state, final_state, -1, 0.0)
            arcs.append(final_arc)
            arcs.append(f"{final_state}")
            sorted_arcs = sorted(arcs, key=lambda a: int(a.split()[0]))

            transcript_fsa = k2.Fsa.from_str("\n".join(sorted_arcs))
            transcript_fsa = k2.arc_sort(transcript_fsa)
            transcript_fsa_list.append(transcript_fsa)

        transcript_fsa_vec = k2.create_fsa_vec(transcript_fsa_list)

        return transcript_fsa_vec
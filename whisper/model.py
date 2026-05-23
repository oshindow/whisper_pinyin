import base64
import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function

try:
    from torch.nn.functional import scaled_dot_product_attention

    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False

from .classifier import ReversalClassifier

class PredHead(nn.Module): #Add commentMore actions
    """Custom CTC head for ASR."""

    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 280,
        num_layers: int = 2,
        use_layer_norm: bool = True,
        use_gelu: bool = True,
        feats_size: int = 256
    ):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            if i == num_layers - 2:
                layers.extend(
                    [
                        nn.Linear(hidden_size, feats_size),
                        nn.GELU() if use_gelu else nn.ReLU(),
                        nn.LayerNorm(feats_size) if use_layer_norm else nn.Identity(),
                    ]
                )
            else:
                layers.extend(
                    [
                        nn.Linear(hidden_size, hidden_size),
                        nn.GELU() if use_gelu else nn.ReLU(),
                        nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity(),
                    ]
                )
        self.proj = nn.Linear(feats_size, vocab_size)
        # layers.append(nn.Linear(hidden_size, vocab_size))
        self.layers = nn.Sequential(*layers) #Add commentMore actions

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # print("hidden_states.shape", hidden_states.shape)
        feats = self.layers(hidden_states)
        logits = self.proj(feats)
        return feats, logits
    
class CTCHead(nn.Module): #Add commentMore actions
    """Custom CTC head for ASR."""

    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 280,
        num_layers: int = 2,
        use_layer_norm: bool = True,
        use_gelu: bool = True,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU() if use_gelu else nn.ReLU(),
                    nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity(),
                ]
            )
        # layers.append(nn.Linear(hidden_size, vocab_size))
        self.layers = nn.Sequential(*layers) #Add commentMore actions
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden = self.layers(hidden_states)
        logits = self.proj(hidden)
        return hidden, logits

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int
    n_accents: int = 6


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


@contextmanager
def disable_sdpa():
    prev_state = MultiHeadAttention.use_sdpa
    try:
        MultiHeadAttention.use_sdpa = False
        yield
    finally:
        MultiHeadAttention.use_sdpa = prev_state


class MultiHeadAttention(nn.Module):
    use_sdpa = True

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
            a = scaled_dot_product_attention(
                q, k, v, is_causal=mask is not None and n_ctx > 1
            )
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        return out, qk


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x

class SharedEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, n_accents: int = 6):
        super().__init__()

        # Same convolutional frontend as AudioEncoder
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)

        # Positional encoding (same as AudioEncoder)
        self.register_buffer("positional_embedding", torch.sin(torch.arange(n_ctx).unsqueeze(1) * torch.arange(n_state).unsqueeze(0)))

        # Transformer blocks (same as AudioEncoder)
        self.blocks = nn.ModuleList([ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])
        self.ln_post = nn.LayerNorm(n_state)

        self.accent_classifier = nn.Sequential(
            nn.Linear(n_state, 256),
            nn.ReLU(),
            nn.Linear(256, n_accents)
        )

        self.fsq_project = nn.Linear(2 * n_state, 1927)

        # self.ctc_head = CTCHead()
        # Improved Accent Classification Head
        '''
        self.accent_classifier = nn.Sequential(
            nn.Linear(n_state, 512),  # Increased capacity
            nn.ReLU(),
            nn.LayerNorm(512),  # Normalize activations
            nn.Dropout(0.3),  # Regularization

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3),

            nn.Linear(256, n_accents)  # Final layer (logits for 3 accents)
        )
        '''

        # Projection layer to map Accent ID logits into the same feature space as the encoder output
        self.accent_embedding = nn.Embedding(n_accents, n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            The log-Mel spectrogram of the audio, as used by Whisper.
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (batch_size, n_ctx, n_state)

        assert x.shape[1:] == self.positional_embedding.shape, "Incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)

        # Compute Accent ID logits
        accent_logits = self.accent_classifier(x.mean(dim=1))


        # Project Accent ID logits into the feature space and append to encoder output
        accent_id = accent_logits.argmax(dim=-1)
        accent_features = self.accent_embedding(accent_id).unsqueeze(1)  # (B, 1, D)
        #accent_features = self.accent_embedding(accent_logits).unsqueeze(1)
        audio_features_accent = torch.cat([x, accent_features.expand(-1, x.size(1), -1)], dim=-1)
        
        audio_features_accent = self.vocab_project(audio_features_accent)

        return audio_features_accent, accent_logits

class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)
        


    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x_in = F.gelu(self.conv2(x))
        x = x_in.permute(0, 2, 1)
        # x = x_in
        # print("x shape after conv:", x.shape, "self.positional_embedding shape:", self.positional_embedding.shape)
        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        

        return x, x_in

class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions, ctc_vocab: int = 280, n_accents: int = 6, ctc_layers: int = 2):
        super().__init__()
        self.dims = dims
        self.dims.n_accents = n_accents

        ''' Accent Encoder
        self.encoder = SharedEncoder(
            n_mels=self.dims.n_mels,
            n_ctx=self.dims.n_audio_ctx,
            n_state=self.dims.n_audio_state,
            n_head=self.dims.n_audio_head,
            n_layer=self.dims.n_audio_layer,
            n_accents=self.dims.n_accents
        )
        '''
        
        # ''' Original Encoder
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        # '''
        # self.proj_layer = Linear(self.dims.n_audio_state, ctc_vocab)
        self.ctc_head = CTCHead(vocab_size=ctc_vocab, hidden_size=self.dims.n_audio_state, num_layers=ctc_layers)
        self.stct_head = CTCHead(vocab_size=7, hidden_size=self.dims.n_audio_state, num_layers=ctc_layers)

        self.accent_classifier = nn.Sequential(
            nn.Linear(self.dims.n_audio_state, 256),
            nn.ReLU(),
            nn.Linear(256, n_accents)
        )
        self.acc_head = PredHead(vocab_size=13, hidden_size=self.dims.n_audio_state, num_layers=2, feats_size=256)
        # self.fsq_project = nn.Linear(self.dims.n_audio_state, 500)
        
        
        self.accent_embedding = nn.Embedding(n_accents, self.dims.n_audio_state)

        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )

        ### **New Projection Layer** to map input with accent features back to original feature size
        self.decoder_projection = nn.Linear(
            self.dims.n_audio_state + self.dims.n_audio_state,  # Expanded due to accent embeddings
            self.dims.n_audio_state  # Map back to original dimension
        )
        ###
        self.spk_grl = ReversalClassifier(
                            input_dim=768,
                            hidden_dim=256,
                            output_dim=368,
                            gradient_clipping_bounds=1.0)
        
        

        # use the last half among the decoder layers for time alignment by default;
        # to use a specific set of heads, see `set_alignment_heads()` below.
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True

        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)
        alignment_heads_dense = self.get_buffer("alignment_heads").to_dense()
        self.register_buffer("alignment_heads", alignment_heads_dense, persistent=False)
        # self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)
        alignment_heads_dense = self.get_buffer("alignment_heads").to_dense()
        self.register_buffer("alignment_heads", alignment_heads_dense, persistent=False)
    
    ### Accent System 
    def embed_audio(self, mel: torch.Tensor):
        projected_features, _ = self.encoder(mel)
        return projected_features
    
    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        print('whisper model logits function')
        return self.decoder(tokens, audio_features)

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor):
        print('whisper model forward function')
        projected_features, accent_logits = self.encoder(mel)
        print(f"Encoder output shape (before projection): {projected_features.shape}")

        # **Fix: Ensure decoder receives properly sized features**
        projected_features = self.decoder_projection(projected_features)

        print(f"Encoder output shape (after projection): {projected_features.shape}")
        logits = self.decoder(tokens, projected_features)
        return logits, accent_logits
    ###
    ''' Original code
    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)
    
    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)
    
    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        print("using local forward")
        return self.decoder(tokens, self.encoder(mel))
    '''
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function

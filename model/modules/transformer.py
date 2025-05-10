import torch
from dataclasses import dataclass
from .embedding import Embedding
from .transformer_block import TransformerBlock
from .rope_llama import *
from .rms_norm import RMSNorm
from .linear import Linear
import torch.nn as nn
import torch.nn.functional as F

class Transformer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size=config.vocab_size
        self.num_layer=config.num_layer
        self.seq_len=config.seq_len
        self.tok_embed=Embedding(config.vocab_size,config.d_model)
        self.layers=nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layer))
        self.norm=RMSNorm(config)
        self.lm_head=Linear(config.d_model,config.vocab_size)
        freqs_cis = precompute_freqs_cis(
            config.d_model// config.num_heads,
            config.seq_len * 2,
        )
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self,ids,targets=None):
        _,T=ids.size()
        x=self.tok_embed(ids)
        freqs_cis = self.freqs_cis[:T]
        #print(f"T: {T}, freqs_cis shape: {freqs_cis.shape}")
        for layer in self.layers:
            x=layer(x,freqs_cis)
        x=self.norm(x)
        logits=self.lm_head(x).float()
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
import torch
from dataclasses import dataclass
from modules.embedding import Embedding
from modules.transformer_block import TransformerBlock
from modules.rms_norm import RMSNorm
from modules.linear import Linear
import torch.nn as nn

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

    def forward(self,ids):
        x=self.tok_embed(ids)
        for layer in self.layers:
            x=layer(x)
        x=self.norm(x)
        logits=self.lm_head(x).float()
        return logits
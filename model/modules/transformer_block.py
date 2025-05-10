import torch
from .multihead_attention import MultiHeadAttention
from .rms_norm import RMSNorm
from .swiglu import SWIGLU


class TransformerBlock(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.d_model=config.d_model
        self.num_heads=config.num_heads
        self.norm=RMSNorm(config)
        self.mha=MultiHeadAttention(config)
        self.swiglu=SWIGLU(config)
    def forward(self,x):
        mha_output=x+self.mha(self.norm(x))
        out=mha_output+self.swiglu(self.norm(mha_output))
        return out

from dataclasses import dataclass

@dataclass
class GPTConfig:
    seq_len: int = 1024 
    vocab_size: int = 50257 
    n_layer: int = 12 
    num_heads: int = 12 
    d_model: int = 768
    d_ff:int=None
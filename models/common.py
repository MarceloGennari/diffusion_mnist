"""
Marcelo Gennari do Nascimento, 2022
marcelogennari@outlook.com

Code for Sinusoidal Positional Embedding was derived from:
https://huggingface.co/blog/annotated-diffusion
"""

import torch
import math


class SinusoidalPositionEmbeddings(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TemporalEmbedding(torch.nn.Module):
    def __init__(self, dim_emb: int, dim_out: int):
        super().__init__()
        self.linear = torch.nn.Linear(dim_emb, dim_out)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        emb = self.linear(temb)
        emb = emb[:, :, None, None]
        out = x + emb
        return out

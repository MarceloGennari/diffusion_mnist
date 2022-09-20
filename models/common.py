"""
Marcelo Gennari do Nascimento, 2022
marcelogennari@outlook.com

Code for SinusoidalPositionEmbedding and TemporalEmbedding 
was derived from https://huggingface.co/blog/annotated-diffusion

Code for LinearAttention and Layer Norm was derived from:
https://github.com/lucidrains/denoising-diffusion-pytorch
"""

import torch
import math
from einops import rearrange, reduce


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
        self.temb = SinusoidalPositionEmbeddings(dim_emb)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        temb = self.temb(time)
        emb = self.linear(temb)
        emb = emb[:, :, None, None]
        out = x + emb
        return out


class LayerNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = torch.nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class LinearAttention(torch.nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class CrossAttention(torch.nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.dim_head = dim_head

        self.label_embedding = torch.nn.Embedding(10, dim)

        self.to_qv = torch.nn.Conv2d(dim, hidden_dim * 2, 1, bias=False)
        self.to_k = torch.nn.Linear(dim, hidden_dim)

        self.to_out = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x: torch.Tensor, label: torch.Tensor):
        b, c, h, w = x.shape
        qv = self.to_qv(x).chunk(2, dim=1)
        q, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qv)

        label_embedding = self.label_embedding(label)
        k = self.to_k(label_embedding)
        k = k.view(b, self.heads, self.dim_head).unsqueeze(-1)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

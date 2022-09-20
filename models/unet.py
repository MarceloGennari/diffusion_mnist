"""
Marcelo Gennari do Nascimento, 2022
marcelogennari@outlook.com

This is my version of a UNet. Not 100% sure that this is correct,
or if the temporal embedding are correct as well. It looks like it
works fine, even though might be a bit of an overly large model.
"""

import torch
from torch import nn
from .common import SinusoidalPositionEmbeddings, TemporalEmbedding


class ConvGroupNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        batch1 = nn.BatchNorm2d(out_channels)
        relu1 = nn.ReLU()

        conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        batch2 = nn.BatchNorm2d(out_channels)
        relu2 = nn.ReLU()

        self.feat = nn.Sequential(conv1, batch1, relu1, conv2, batch2, relu2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feat(x)


class UNet(nn.Module):
    def __init__(self, dim_emb: int = 128):
        ch = [32, 64, 64, 32]
        super().__init__()
        # Positional Embedding
        self.temb = SinusoidalPositionEmbeddings(dim_emb)
        self.embedding1 = TemporalEmbedding(dim_emb, 1)

        # Input is 1x28x28
        self.block1 = ConvGroupNorm(1, ch[0])
        self.down1 = nn.Conv2d(ch[0], ch[0], 4, stride=2, padding=1)

        # Now input is 32x14x14
        self.embedding2 = TemporalEmbedding(dim_emb, ch[0])
        self.block2 = ConvGroupNorm(ch[0], ch[1])
        self.down2 = nn.Conv2d(ch[1], ch[1], 4, stride=2, padding=1)

        # Now input is 64x7x7
        self.embedding3 = TemporalEmbedding(dim_emb, ch[1])
        self.block3 = ConvGroupNorm(ch[1], ch[2])
        self.up1 = nn.ConvTranspose2d(ch[2], ch[2], 4, stride=2, padding=1)

        # Now input is 64x14x14
        new_ch = ch[2] + ch[1]
        self.embedding4 = TemporalEmbedding(dim_emb, new_ch)
        self.block4 = ConvGroupNorm(new_ch, ch[3])
        self.up2 = nn.ConvTranspose2d(ch[3], ch[3], 4, stride=2, padding=1)

        # Now input is 16x28x28
        new_ch = ch[3] + ch[0]
        self.embedding5 = TemporalEmbedding(dim_emb, new_ch)
        self.block5 = ConvGroupNorm(new_ch, 1)
        self.out = nn.Conv2d(1, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x0 = self.embedding1(x, self.temb(t))
        x1 = self.block1(x0)
        x1 = self.embedding2(x1, self.temb(t))
        x2 = self.block2(self.down1(x1))
        x2 = self.embedding3(x2, self.temb(t))
        x3 = self.up1(self.block3(self.down2(x2)))
        x4 = torch.cat([x2, x3], dim=1)
        x4 = self.embedding4(x4, self.temb(t))
        x5 = self.up2(self.block4(x4))
        x6 = torch.cat([x5, x1], dim=1)
        x6 = self.embedding5(x6, self.temb(t))
        out = self.out(self.block5(x6))
        return out

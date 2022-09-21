"""
Marcelo Gennari do Nascimento, 2022
marcelogennari@outlook.com

This script performs sampling by using the DDIM algorithm, which is a
deterministic solution to the generation of images:
https://arxiv.org/abs/2010.02502

It is supposed to be 10x to 100x quicker, and it uses the same training 
procedure as DDPM, so the trained model can be reused here
"""

from tqdm import trange

import torch
from models import UNet
from diffusion_model import DiffusionProcess

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Prepare model
    device = "cpu"
    batch_size = 100
    model = UNet().to(device)
    model.load_state_dict(torch.load("unet_mnist.pth"))
    process = DiffusionProcess()

    # Sampling
    xt = torch.randn(batch_size, 1, 28, 28)

    model.eval()
    with torch.no_grad():
        for t in trange(999, -1, -1):
            time = torch.ones(batch_size) * t
            et = model(xt.to(device), time.to(device))  # predict noise
            xt = process.inverse_DDIM(xt, et.cpu(), t)

    labels = ["Generated Images"] * 9

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(xt[i][0], cmap="gray", interpolation="none")
        plt.title(labels[i])
    plt.show()

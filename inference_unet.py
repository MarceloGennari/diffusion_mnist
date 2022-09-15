"""
Marcelo Gennari do Nascimento, 2022
marcelogennari@outlook.com

This script performs the sampling given the trained UNet model
"""
from tqdm import trange

import torch
from models import UNet
from diffusion_model import DiffusionProcess

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Prepare model
    batch_size = 100
    model = UNet()
    model.load_state_dict(torch.load("unet_mnist.pth"))
    process = DiffusionProcess()

    # Sampling
    xt = torch.randn(batch_size, 1, 28, 28)

    model.eval()
    with torch.no_grad():
        for t in trange(999, -1, -1):
            time = torch.ones(batch_size) * t
            et = model(xt, time)  # predict noise
            xt = process.inverse(xt, et, t)

    labels = ["Generated Images"] * 9

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(xt[i][0], cmap="gray", interpolation="none")
        plt.title(labels[i])
    plt.show()

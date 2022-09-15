"""
Marcelo Gennari do Nascimento, 2022
marcelogennari@outlook.com
"""

import torch
from mnist_dataset import get_mnist_dataloader
from diffusion_model import DiffusionProcess

import matplotlib.pyplot as plt

from models.unet import UNet

if __name__ == "__main__":
    # prepare images
    trainloader, testloader = get_mnist_dataloader()
    idx, (images, labels) = next(enumerate(testloader))

    model = UNet()

    # Plot original images
    t = torch.randint(0, 1000, (images.shape[0],))
    images = model(images, t)
    print(images.shape)

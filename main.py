"""
Marcelo Gennari do Nascimento, 2022
marcelogennari@outlook.com
"""

import torch
from mnist_dataset import get_mnist_dataloader
from diffusion_model import DiffusionProcess

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # prepare images
    trainloader, testloader = get_mnist_dataloader()
    idx, (images, labels) = next(enumerate(testloader))

    # Plot original images
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(images[i][0], cmap="gray", interpolation="none")
        plt.title(labels[i].item())
    plt.show()

    # Prepare diffusion steps
    time_step = torch.randint(0, 1000, (images.shape[0],))
    noise = torch.randn(images.shape)
    diffusion = DiffusionProcess()
    diffused_images = diffusion.forward(images, time_step, noise)

    # Plot Noisy images
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(diffused_images[i][0], cmap="gray", interpolation="none")
        plt.title(f"Label {labels[i].item()} at step {time_step[i]}")
    plt.show()

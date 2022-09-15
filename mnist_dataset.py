"""
Marcelo Gennari do Nascimento, 2022
marcelogennari@outlook.com

This script has been adapted from https://nextjournal.com/gkoehler/pytorch-mnist
"""

from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST
from typing import Tuple


def get_mnist_dataloader(batch_size: int = 256) -> Tuple[DataLoader, DataLoader]:
    """
    Convenient helper function to get the loaders for the mnist dataset
    Args:
        batch_size (int): the size of batches

    Returns:
        Tuple[DataLoader, DataLoader]: train loader and test loader respectively
    """
    to_tensor = torchvision.transforms.ToTensor()
    normalize = torchvision.transforms.Normalize((0.1307,), (0.3081,))
    transform = torchvision.transforms.Compose([to_tensor, normalize])

    trainset = MNIST("./data/", train=True, download=True, transform=transform)
    testset = MNIST("./data/", train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

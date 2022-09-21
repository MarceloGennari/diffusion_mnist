"""
Marcelo Gennari do Nascimento, 2022
marcelogennari@outlook.com

This script is used to train the UNet to predict the noise at different
timestamps of the diffusion process. The loss function is a simple mean
squared error between the actual noise and the predicted noise based on
the diffused image, according to the original paper: 
https://arxiv.org/pdf/2006.11239.pdf
"""

from tqdm import tqdm, trange

import torch
from torch import optim
from mnist_dataset import get_mnist_dataloader
from diffusion_model import DiffusionProcess

from models import ConditionalUNet

if __name__ == "__main__":
    # Prepare images
    trainloader, testloader = get_mnist_dataloader()
    idx, (images, labels) = next(enumerate(testloader))

    # Prepare model and training
    device = "cpu"
    model = ConditionalUNet().to(device)
    process = DiffusionProcess()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 80)
    criterion = torch.nn.MSELoss()

    # Training Loop
    epochs = 100
    for e in trange(epochs):
        running_loss = 0
        progress_bar = tqdm(trainloader, leave=False)
        for image, label in progress_bar:
            # Sampling t, epsilon, and diffused image
            t = torch.randint(0, 1000, (image.shape[0],))
            epsilon = torch.randn(image.shape)
            diffused_image = process.forward(image, t, epsilon)

            # Backprop
            optimizer.zero_grad()
            output = model(diffused_image.to(device), t.to(device), label.to(device))
            loss = criterion(epsilon.to(device), output)
            loss.backward()
            optimizer.step()
            loss_value = loss.cpu().item()
            running_loss += loss_value
            progress_bar.set_description(f"Loss: {loss_value:.4f}")

        scheduler.step()

        # Save model after every epoch
        torch.save(model.state_dict(), "unet_mnist.pth")

        # Logging results
        running_loss /= len(trainloader)
        tqdm.write(f"Mean loss for Epoch {e + 1}: {running_loss:.4f}")

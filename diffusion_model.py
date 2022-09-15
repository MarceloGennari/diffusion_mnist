"""
Marcelo Gennari do Nascimento, 2022
marcelogennari@outlook.com
"""
import torch
from torch import Tensor
from typing import List


class DiffusionProcess:
    """
    Class that implements the forward and inverse diffusion process according to
    the DDPM paper: https://arxiv.org/pdf/2006.11239.pdf

    Attributes:
        variance_schedule (torch.Tensor): list with the variance value at each
            timestep according to DDPM paper
        alpha (torch.Tensor): list of "complement" values for variance defined
            in the DDPM paper. It is the same as 1-variance_schedule
        alpha_bar (torch.Tensor): cummulative product defined int he DDPM
            paper above. It is derived directly from the variance schedule
    """

    def __init__(self, variance_schedule: List[float] = None) -> None:
        """
        Args:
            variance_schedule (list): list with the variance value at each
                timestep according to DDPM paper. If left None, it will default
                to a list of linearly increasing variance from 1e-4 to 0.02 in
                1000 steps
        """
        if variance_schedule is None:
            variance_schedule = torch.linspace(1e-4, 0.01, steps=1000)
        self.variance_schedule = Tensor(variance_schedule)
        self.alpha = 1 - self.variance_schedule
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward(self, x_0: Tensor, time_step: Tensor, noise: Tensor) -> Tensor:
        """
        This applies the forward diffusion process to original image ``x`` to
        timestamp ``time_step``, using a sample ``noise`` from a zero mean, unit
        variance gaussian distribution. The formula for the forward propagation,
        given step time_step and original tensor x is:
            p(x_t | x_0) = N(sqrt(alpha_t) * x_0, (1-alpha_t) * I)
        where I is the identity matrix. This can be reparameterised as following:
            p(x_t | x_0) = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * noise
        where ``noise``~N(0, I). This is the expression used in this function.

        Args:
            x_0 (torch.Tensor): original image. It can be of any shape, since
                diffusion is independent and identically distributed (iid)
            time_step (torch.Tensor): which step to diffuse the original image.
                It is a Tensor with numbers between 0 and len(self.alpha_bar). It
                also has to have the same batch size as ``x_0``
            noise (torch.Tensor): the noise to be added at this ``time_step``. It
                has to be a tensor sampled from a zero-mean unit-variance normal
                distribution.

        Returns:
            torch.Tensor: the result of diffusig original image ``x_0`` to
                ``time_step`` using the variance schedule :attr:alpha_bar
        """
        # Checking for validity of input
        assert torch.all(time_step >= 0).item()
        assert torch.all(time_step < len(self.alpha_bar)).item()
        assert time_step.shape[0] == x_0.shape[0]
        std_dev = torch.sqrt(1 - self.alpha_bar[time_step])
        mean_multiplier = torch.sqrt(self.alpha_bar[time_step])

        # This makes sure that variance and mean multiplier are both broadcastable
        std_dev = std_dev[:, None, None, None].to(x_0.device)
        mean_multiplier = mean_multiplier[:, None, None, None].to(x_0.device)

        diffused_images = mean_multiplier * x_0 + std_dev * noise
        return diffused_images

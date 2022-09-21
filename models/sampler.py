"""
Marcelo Gennari do Nascimento, 2022
marcelogennari@outlook.com
"""
import torch
from torch import nn, Tensor
from typing import List


class DDPMSampler(nn.Module):
    def __init__(self, model: nn.Module, variance_schedule: List[float] = None) -> None:
        super().__init__()
        self.model = model
        if variance_schedule is None:
            variance_schedule = torch.linspace(1e-4, 0.01, steps=1000)
        self.variance_schedule = Tensor(variance_schedule)
        self.alpha = 1 - self.variance_schedule
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def inverse(self, xt: Tensor, et: Tensor, t: int) -> Tensor:
        """
        This applies the unconditional sampling of the diffusion step. It uses the
        equation as follow:
            p(x_{t-1}| x_t) = mu_t + std_dev_t * N(0, I)
            mu_t = (1/sqrt(alpha_t)) * (xt - noise_scale * et)
            noise_scale = (1-alpha_t) / sqrt(1-alpha_bar_t)
            std_dev_t = sqrt(variance_schedule)
        this is from the DDPM paper.

        Args:
            xt (torch.Tensor): noisy image at time ``t``.
            et (torch.Tensor): predicted error from diffusion model, which is
                usually the output of the trained UNet architecture
            t (int): the time ``t`` of the diffusion process

        Returns:
            torch.Tensor: the result of the sampling x_{t-1}
        """
        scale = 1 / torch.sqrt(self.alpha[t])
        noise_scale = (1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_bar[t])
        std_dev = torch.sqrt(self.variance_schedule[t])
        mu_t = scale * (xt - noise_scale * et)

        z = torch.randn(xt.shape) if t > 1 else torch.Tensor([0])
        xt = mu_t + std_dev * z  # remove noise from image
        return xt

    def forward(self, xt: Tensor, t: int) -> Tensor:
        time = torch.ones(xt.shape[0]) * t
        et = self.model(xt, time)
        xt = self.inverse(xt, et, t)
        return xt



class DDIMSampler(DDPMSampler):
    def __init__(self, model: nn.Module, variance_schedule: List[float] = None) -> None:
        super().__init__(model, variance_schedule)

    def inverse(self, xt: Tensor, et: Tensor, t: int) -> Tensor:
        """
        This applies the unconditional sampling of the diffusion step using the
        DDIM method: https://arxiv.org/abs/2010.02502
        f_theta acts as an approximation for x_0, and the rest follows equation
        (7) in the paper. For DDIM, we have that std_dev = 0
        This solves the problem of stochasticity, and it is supposed to be 10x
        to 100x quicker than the DDPM method
        """
        den = 1 / torch.sqrt(self.alpha_bar[t])
        f_theta = (xt - torch.sqrt(1 - self.alpha_bar[t]) * et) * den
        if t > 0:
            part1 = torch.sqrt(self.alpha_bar[t - 1]) * f_theta
            part2 = torch.sqrt(1 - self.alpha_bar[t - 1])
            den = 1 / torch.sqrt(1 - self.alpha_bar[t])
            scale = (xt - torch.sqrt(self.alpha_bar[t]) * f_theta) * den
            xt = part1 + part2 * scale
        else:
            xt = f_theta

        return xt
    

class DDPMConditionalSampler(DDPMSampler):
    def __init__(self, model: nn.Module, variance_schedule: List[float] = None) -> None:
        super().__init__(model, variance_schedule)

    def forward(self, xt: Tensor, t: int, label: int) -> Tensor:
        time = torch.ones(xt.shape[0]) * t
        label = (torch.ones(xt.shape[0]) * label).long()
        et = self.model(xt, time, label)
        xt = self.inverse(xt, et, t)
        return xt

class DDIMConditionalSampler(DDPMConditionalSampler):
    def __init__(self, model: nn.Module, variance_schedule: List[float] = None) -> None:
        super().__init__(model, variance_schedule)

    def inverse(self, xt: Tensor, et: Tensor, t: int) -> Tensor:
        """
        This applies the unconditional sampling of the diffusion step using the
        DDIM method: https://arxiv.org/abs/2010.02502
        f_theta acts as an approximation for x_0, and the rest follows equation
        (7) in the paper. For DDIM, we have that std_dev = 0
        This solves the problem of stochasticity, and it is supposed to be 10x
        to 100x quicker than the DDPM method
        """
        den = 1 / torch.sqrt(self.alpha_bar[t])
        f_theta = (xt - torch.sqrt(1 - self.alpha_bar[t]) * et) * den
        if t > 0:
            part1 = torch.sqrt(self.alpha_bar[t - 1]) * f_theta
            part2 = torch.sqrt(1 - self.alpha_bar[t - 1])
            den = 1 / torch.sqrt(1 - self.alpha_bar[t])
            scale = (xt - torch.sqrt(self.alpha_bar[t]) * f_theta) * den
            xt = part1 + part2 * scale
        else:
            xt = f_theta

        return xt
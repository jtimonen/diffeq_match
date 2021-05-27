import torch
import torch.nn as nn


def tensor_to_numpy(x: torch.Tensor):
    return x.cpu().detach().numpy()


def add_noise(x: torch.Tensor, sigma):
    """Add normally distributed noise to Tensor."""
    if sigma > 0:
        x = x + sigma * torch.randn_like(x)
    return x


def num_trainable_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

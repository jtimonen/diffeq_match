import torch


def tensor_to_numpy(x: torch.Tensor):
    return x.cpu().detach().numpy()


def add_noise(x: torch.Tensor, sigma):
    """Add normally distributed noise to Tensor."""
    if sigma > 0:
        x = x + sigma * torch.randn_like(x)
    return x

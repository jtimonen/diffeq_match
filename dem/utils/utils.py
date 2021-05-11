import torch


def tensor_to_numpy(x: torch.Tensor):
    return x.cpu().detach().numpy()

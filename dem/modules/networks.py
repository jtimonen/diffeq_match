import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func


class TanhNetOneLayer(nn.Module):
    """A fully connected network with one hidden layer. Uses the hyperbolic
    tangent activation function.
    """

    def __init__(self, n_input: int, n_output: int, n_hidden: int = 128):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.log_R = torch.nn.Parameter(torch.tensor(-0.5).float(), requires_grad=True)
        self.layers = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Tanh(), nn.Linear(n_hidden, n_output)
        )

    def forward(self, z: torch.Tensor):
        """Pass the tensor z through the network."""
        y = self.layers(z)
        R = torch.exp(self.log_R)
        return R * func.normalize(y, dim=1)


class ReluNetOne(nn.Module):
    """Network with one output node."""

    def __init__(self, n_input: int, n_hidden: int = 128):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.layers = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1)
        )

    def forward(self, z: torch.Tensor):
        """Pass the tensor z through the network."""
        y = 0.3 * torch.sigmoid(self.layers(z))
        return y


class TanhNetTwoLayer(nn.Module):
    """A fully connected network with two hidden layers. Uses the hyperbolic
    tangent activation function.
    """

    def __init__(self, n_input: int, n_output: int, n_hidden: int = 32):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        # self.log_R = torch.nn.Parameter(torch.tensor(-0.5).float(), requires_grad=True)
        self.layers = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_output),
        )

    def forward(self, z: torch.Tensor):
        """Pass the tensor z through the network."""
        y = self.layers(z)
        # R = torch.exp(self.log_R)
        # theta = y[:, 0]
        # z1 = R * torch.cos(theta)
        # z2 = R * torch.sin(theta)
        # z = torch.vstack((z1, z2)).T
        return y  # R * func.normalize(y, dim=1)


class LeakyReluNetTwoLayer(nn.Module):
    """A fully connected network with two hidden layers. Uses the ReLU
    activation function.
    """

    def __init__(self, n_input: int, n_output: int, n_hidden: int = 64):
        super().__init__()
        self.n_input = n_input
        self.n_input = n_output
        self.n_hidden = n_hidden
        self.layers = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n_hidden, n_output),
        )

    def forward(self, z: torch.Tensor):
        """Pass the tensor z through the network."""
        return self.layers(z)


class Reverser(nn.Module):
    """Reverses the sign of module output."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.n_input = module.n_input

    def forward(self, x: torch.Tensor):
        return -self.module(x)


class ConstantLinear(nn.Module):
    """Linear operation Ax+b where A and b are constant.

    :param weight: the matrix A, shape (out_features,in_features)
    :param bias: vector b, shape (out_features)
    """

    def __init__(self, weight: np.ndarray, bias: np.ndarray):
        super().__init__()
        self.weight = torch.from_numpy(weight).float()
        self.bias = torch.from_numpy(bias).float()
        self.n_input = weight.shape[1]
        self.n_output = weight.shape[0]

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.linear(x, weight=self.weight, bias=self.bias)

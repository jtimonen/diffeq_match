import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func


def create_sequential(
    n_input: int, n_output: int, n_hidden: int, n_hidden_layers: int, activation
):
    if n_hidden_layers == 1:
        out = nn.Sequential(
            nn.Linear(n_input, n_hidden), activation, nn.Linear(n_hidden, n_output)
        )
    elif n_hidden_layers == 2:
        out = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            activation,
            nn.Linear(n_hidden, n_hidden),
            activation,
            nn.Linear(n_hidden, n_output),
        )
    else:
        raise ValueError("n_hidden_layers should be 1 or 2!")
    return out


class MultiLayerNet(nn.Module):
    """A fully connected network with one hidden layer. Uses the hyperbolic
    tangent activation function.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int = 128,
        n_hidden_layers: int = 1,
        activation=nn.Tanh,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.n_hidden_layers = n_hidden_layers
        self.act_name = str(activation)
        self.layers = create_sequential(
            n_input, n_output, n_hidden, n_hidden_layers, activation
        )

    def forward(self, x: torch.Tensor):
        """Pass the tensor x through the network."""
        return self.layers(x)

    def __repr__(self):
        n1, n2 = self.n_hidden, self.n_hidden_layers
        desc = "MultiLayerNet(n_hidden=%d, n_hidden_layers=%d" % (n1, n2)
        desc += ", activation=" + self.act_name + ")"
        return desc


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

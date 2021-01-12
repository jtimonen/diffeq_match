import torch
import torch.nn as nn


class TanhNetOneLayer(nn.Module):
    """A fully connected network with one hidden layer. Uses the hyperbolic
    tangent activation function.
    """

    def __init__(self, n_input: int, n_output: int, n_hidden: int = 32):
        super().__init__()
        self.n_input = n_input
        self.n_input = n_output
        self.n_hidden = n_hidden
        self.layers = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Tanh(), nn.Linear(n_hidden, n_output)
        )

    def forward(self, z: torch.Tensor):
        """Pass the tensor z through the network."""
        return self.layers(z)


class ReluNetTwoLayer(nn.Module):
    """A fully connected network with two hidden layers. Uses the ReLU
    activation function.
    """

    def __init__(self, n_input: int, n_output: int, n_hidden: int = 32):
        super().__init__()
        self.n_input = n_input
        self.n_input = n_output
        self.n_hidden = n_hidden
        self.layers = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )

    def forward(self, z: torch.Tensor):
        """Pass the tensor z through the network."""
        return self.layers(z)

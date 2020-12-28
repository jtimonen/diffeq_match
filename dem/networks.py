import torch
import torch.nn as nn


class PseudotimeEncoder(nn.Module):
    """Encodes latent representation to latent time."""

    def __init__(self, D: int, H: int = 128, eps=0.001, momentum=0.01, p_dropout=0.1):
        super().__init__()
        self.D = D
        self.H = H
        self.layer1 = nn.Sequential(
            nn.Linear(D, H),
            nn.BatchNorm1d(num_features=H, eps=eps, momentum=momentum),
            nn.ReLU(),
            nn.Dropout(p_dropout),
        )
        self.layer2 = nn.Linear(H, 1)

    def forward(self, z: torch.Tensor):
        """Forward pass."""
        h = self.layer1(z)
        t = self.layer2(h)
        return torch.sigmoid(t)

    @torch.no_grad()
    def forward_numpy(self, z):
        """Forward pass but using numpy arrays."""
        zt = torch.from_numpy(z).float()
        tt = self.forward(zt)
        return tt.detach().cpu().numpy()


class Discriminator(nn.Module):
    """Two-class discriminator network."""

    def __init__(
        self,
        n_input,
        n_hidden,
        p_dropout: float = 0.1,
        momentum: float = 0.01,
        eps: float = 0.001,
    ):
        super().__init__()
        self.n_classes = 2
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.layer1 = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.BatchNorm1d(n_hidden, momentum=momentum, eps=eps),
            nn.ReLU(),
            nn.Dropout(p_dropout),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden, momentum=momentum, eps=eps),
            nn.ReLU(),
            nn.Dropout(p_dropout),
        )
        self.layer3 = nn.Linear(n_hidden, self.n_classes)
        self.layer4 = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss(reduction="none")

    def forward(self, w: torch.Tensor, z: torch.Tensor):
        """Forward pass. Outputs probability for class 0."""
        x = torch.cat((w, z), dim=1)
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        logp = self.layer4(h)
        return logp


class TanhNetOneLayer(nn.Module):
    """A fully connected network with one hidden layer. Uses Tanh activation
    functions and batch normalization.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int = 64,
        eps=0.001,
        momentum=0.01,
        multiplier=1.0,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_input = n_output
        self.hidden_dim = n_hidden
        self.multiplier = multiplier
        self.layer1 = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.BatchNorm1d(num_features=n_hidden, eps=eps, momentum=momentum),
            nn.Tanh(),
            nn.Dropout(0.1),
        )
        self.layer2 = nn.Linear(n_hidden, n_output)
        self.log_magnitude = torch.nn.Parameter(
            -0.5 + 0.1 * torch.randn(1), requires_grad=True
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor):
        """Pass the tensor z through the network."""
        h = self.layer1(z)
        h = self.layer2(h)
        f = self.layer3(h)
        f = nn.functional.normalize(f, dim=1)
        return torch.exp(self.log_magnitude) * f


class TanhNetTwoLayer(nn.Module):
    """A fully connected network with two hidden layers. Uses Tanh activation
    functions and batch normalization.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int = 64,
        eps=0.001,
        momentum=0.01,
        multiplier=1.0,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_input = n_output
        self.hidden_dim = n_hidden
        self.multiplier = multiplier
        self.layer1 = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.BatchNorm1d(num_features=n_hidden, eps=eps, momentum=momentum),
            nn.Tanh(),
            nn.Dropout(0.1),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(num_features=n_hidden, eps=eps, momentum=momentum),
            nn.Tanh(),
            nn.Dropout(0.1),
        )
        self.layer3 = nn.Linear(n_hidden, n_output)
        self.log_magnitude = torch.nn.Parameter(
            -0.5 + 0.1 * torch.randn(1), requires_grad=True
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor):
        """Pass the tensor z through the network."""
        h = self.layer1(z)
        h = self.layer2(h)
        f = self.layer3(h)
        f = nn.functional.normalize(f, dim=1)
        return torch.exp(self.log_magnitude) * f

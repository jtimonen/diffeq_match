import torch
import torch.nn as nn
from scdyn.utils.utils import assert_numeric


class Encoder(nn.Module):
    """Encodes normalized expression data to latent cell representations. Has one hidden layer.
    Includes batch normalization, ReLU activation and Dropout.

    :param G: input dimension (number of genes)
    :type G: int
    :param D: output dimension
    :type D: int
    :param H: hidden layer dimension
    :type H: int
    :param var_lower_bound: lower bound for the returned variance
    :type var_lower_bound: float
    :param eps: *eps* parameter of *BatchNorm1d*
    :type eps: float
    :param momentum: *momentum* parameter of *BatchNorm1d*
    :type momentum: float
    :param p_dropout: proportion parameter of *torch.nn.Dropout*
    :type p_dropout: float
    """

    def __init__(
        self,
        G: int,
        D: int,
        H: int,
        var_lower_bound: float = 1e-4,
        eps=0.001,
        momentum=0.01,
        p_dropout=0.1,
    ):
        super().__init__()
        self.G = G
        self.D = D
        self.H = H
        self.layer1 = nn.Sequential(
            nn.Linear(G, H),
            nn.BatchNorm1d(num_features=H, eps=eps, momentum=momentum),
            nn.ReLU(),
            nn.Dropout(p_dropout),
        )
        self.layer2_mean = nn.Linear(H, D)
        self.layer2_logvar = nn.Linear(H, D)
        assert var_lower_bound >= 0, "variance lower bound must be non-negative"
        self.var_lower_bound = var_lower_bound

    def forward(self, x: torch.Tensor):
        """Pass the tensor x through the network.

        :param x: tensor with shape *[n_minibatch, G]*
        :type x: torch.Tensor
        :return: Means and variances, both with shape *[n_minibatch, D]*.
        :rtype: *(torch.Tensor, torch.Tensor, torch.Tensor)*
        """
        # Encode
        h = self.layer1(x)
        z_mu = self.layer2_mean(h)
        z_s2 = torch.exp(self.layer2_logvar(h)) + self.var_lower_bound
        return z_mu, z_s2


class TimeEncoder(nn.Module):
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


class TimeEncoderSU(nn.Module):
    """Encodes latent representation to latent time."""

    def __init__(self, G: int, H: int = 128, eps=0.001, momentum=0.01, p_dropout=0.1):
        super().__init__()
        self.G = G
        self.H = H
        self.layer1 = nn.Sequential(
            nn.Linear(2 * G, H),
            nn.BatchNorm1d(num_features=H, eps=eps, momentum=momentum),
            nn.ReLU(),
            nn.Dropout(p_dropout),
        )
        self.layer2_mean = nn.Linear(H, 1)
        self.layer2_logvar = nn.Linear(H, 1)
        self.var_lower_bound = 1e-4

    def forward(self, xs: torch.Tensor, xu: torch.Tensor):
        """Forward pass."""
        x = torch.cat((xs, xu), dim=1)
        h = self.layer1(x)
        t_mu = self.layer2_mean(h)
        t_s2 = torch.exp(self.layer2_logvar(h)) + self.var_lower_bound
        return t_mu, t_s2


class Decoder(nn.Module):
    """Decodes latent representations to positive values.
    Has one hidden layer and uses ReLU activation.

    :param G: input dimension
    :type G: int
    :param D: output dimension
    :type D: int
    :param H: hidden layer dimension
    :type H: int
    """

    def __init__(self, D: int, G: int, H: int = 24):
        super().__init__()
        self.D = D
        self.G = G
        self.H = H
        self.layer1 = nn.Sequential(nn.Linear(D, H), nn.ReLU())
        self.layer2 = nn.Linear(H, G)

    def forward(self, z: torch.Tensor):
        """Pass the tensor *z* through the network.

        :param z: tensor with shape *[n_minibatch, D]*
        :type z: torch.Tensor
        :return: data reconstruction, with shape *[n_minibatch, G]*
        :rtype: torch.Tensor
        """
        px = self.layer1(z)
        out = torch.exp(self.layer2(px))
        assert_numeric(out)
        return out


class LinearDecoder(nn.Module):
    """Decodes latent representations to positive values.

    :param G: input dimension
    :type G: int
    :param D: output dimension
    :type D: int
    """

    def __init__(self, D: int, G: int):
        super().__init__()
        self.D = D
        self.G = G
        self.layer1 = nn.Linear(D, G)

    def forward(self, z: torch.Tensor):
        """Pass the tensor *z* through the network.

        :param z: tensor with shape *[n_minibatch, D]*
        :type z: torch.Tensor
        :return: data reconstruction, with shape *[n_minibatch, G]*
        :rtype: torch.Tensor
        """
        px = self.layer1(z)
        out = torch.exp(px)
        assert_numeric(out)
        return out


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

import torch
import torch.nn as nn

from dem.data import PriorInfo, create_prior_info
from dem.modules.vectorfield import Reverser, VectorField
from dem.modules.discriminator import Discriminator, KdeDiscriminator


def create_model(D: int, n_hidden: int = 32, z_init=None, z_terminal=None):
    """Construct a model with some default settings."""
    vector_field = VectorField(D, n_hidden)
    prior_info = create_prior_info(z_init, z_terminal)
    disc = KdeDiscriminator(D)
    model = GenModel(vector_field, prior_info, disc)
    return model


class GenModel(nn.Module):
    """Main model module."""

    def __init__(
        self, vector_field: VectorField, prior_info: PriorInfo, disc: Discriminator
    ):
        super().__init__()
        self.prior_info = prior_info
        self.field = vector_field
        self.field_reverse = Reverser(self.field)
        self.D = vector_field.D
        self.disc = disc

    def set_bandwidth(self, z_data):
        self.kde.set_bandwidth(z_data)

    def forward(self, N: int):
        ts = torch.linspace(0, 1, N).float()
        z_init = self.draw_init(N)
        z_samp = self.traj(z_init, ts, sde=True, forward=True)
        return torch.transpose(z_samp.diagonal(), 0, 1)

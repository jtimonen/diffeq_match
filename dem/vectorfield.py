from .networks import TanhNetTwoLayer
import torch
import torch.nn as nn
import numpy as np
import torchdiffeq
import torchsde


class VectorField(nn.Module):
    def __init__(self, D, n_hidden):
        super().__init__()
        self.D = D
        self.net_f = TanhNetTwoLayer(D, D, n_hidden)

    @property
    def is_stochastic(self):
        return False

    def forward(self, t, y: torch.Tensor):
        return self.f(t, y)

    def f(self, t, y: torch.Tensor):
        return self.net_f(y)

    def f_numpy(self, y: np.ndarray):
        y_t = torch.from_numpy(y).float()
        f_np = self.field.f(0.0, y_t).cpu().detach().numpy()
        return f_np

    def odeint(self, y_init, ts, **kwargs):
        traj = torchdiffeq.odeint_adjoint(self, y_init, ts, **kwargs)
        return traj

    def traj(self, y_init, ts, **kwargs):
        return self.sdeint(y_init, ts, **kwargs)


class StochasticVectorField(VectorField):
    def __init__(self, D, n_hidden):
        super().__init__(D, n_hidden)
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        ln = torch.Tensor([np.log(0.2)]).float()
        self.log_noise = nn.Parameter(ln, requires_grad=True)

    @property
    def is_stochastic(self):
        return True

    @property
    def diffusion_magnitude(self):
        return torch.exp(self.log_noise)

    def g(self, t, y: torch.Tensor):
        g = self.diffusion_magnitude
        return g * torch.ones_like(y)

    @torch.no_grad()
    def g_numpy(self, y: np.ndarray):
        y = torch.from_numpy(y).float()
        g = self.field.g(0, y).cpu().detach().numpy()
        return g[:, 0]

    def sdeint(self, y_init, ts, **kwargs):
        traj = torchsde.sdeint(self.field, y_init, ts, **kwargs)
        return traj

    def traj(self, y_init, ts, **kwargs):
        return self.sdeint(y_init, ts, **kwargs)


class Reverser(nn.Module):
    """Reverses the sign of VectorField output."""

    def __init__(self, f: VectorField):
        super().__init__()
        self.f = f

    def forward(self, t, y):
        return -self.f(t, y)

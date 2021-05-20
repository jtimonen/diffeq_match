from dem.utils.utils import tensor_to_numpy
import torch
import torch.nn as nn
import numpy as np
import torchdiffeq
import torchsde


class VectorField(nn.Module):
    def __init__(self, net_f: nn.Module):
        super().__init__()
        self.D = net_f.n_input
        self.net_f = net_f

    @property
    def is_stochastic(self):
        return False

    def forward(self, t, y: torch.Tensor):
        return self.f(t, y)

    def f(self, t, y: torch.Tensor):
        return self.net_f(y)

    @torch.no_grad()
    def f_numpy(self, y: np.ndarray):
        y_t = torch.from_numpy(y).float()
        f_np = self.f(0.0, y_t)
        return tensor_to_numpy(f_np)

    def odeint(self, y_init, ts, **kwargs):
        return torchdiffeq.odeint_adjoint(self, y_init, ts, **kwargs)


class StochasticVectorField(VectorField):
    def __init__(self, net_f: nn.Module, init_diffusion: float = 0.3):
        super().__init__(net_f)
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        ln = torch.tensor([np.log(init_diffusion)], dtype=torch.float32)
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
        g = self.field.g(0, y)
        return tensor_to_numpy(g[:, 0])

    def sdeint(self, y_init, ts, **kwargs):
        return torchsde.sdeint(self, y_init, ts, **kwargs)

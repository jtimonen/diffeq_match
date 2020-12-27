import torch
import torch.nn as nn
import numpy as np


def create_ode(D: int, n_hidden: int = 32):
    """Create ODE."""
    f = TanhNetTwoLayer(D, D, n_hidden=n_hidden)
    return ODE(f)


class ODE(nn.Module):
    """An ordinary differential equation module.

    :param f: A module defining the right-hand side of the ODE system.
    :type f: nn.Module
    """

    def __init__(self, f: nn.Module):
        super().__init__()
        self.f = f

    def forward(
        self,
        z0: torch.Tensor,
        t: torch.Tensor,
        n_steps: int,
        direction: float = 1.0,
    ):
        """Forward pass using RK4 solver."""
        t = t.view(-1, 1)
        H = t / n_steps
        z = torch.zeros_like(z0)
        z = z + z0
        for i in range(0, n_steps):
            z = z + self.rk4_step(z, H, direction)
        return z

    def solve(
        self,
        z0: torch.Tensor,
        t: torch.Tensor,
        n_steps: int,
        direction: float = 1.0,
    ):
        """Forward pass using RK4 solver, saving intermediate steps."""
        N = z0.shape[0]
        D = z0.shape[1]
        t_flat = t.flatten()
        H_flat = t_flat / n_steps
        H = H_flat.view(-1, 1)
        Z = torch.zeros(n_steps + 1, N, D, device=z0.device).float()
        T = torch.zeros(n_steps + 1, N, device=z0.device).float()
        ttt = torch.zeros_like(t_flat)
        T[0, :] += ttt
        z = torch.zeros_like(z0)
        z = z + z0
        Z[0, :, :] += z0
        for i in range(0, n_steps):
            z = z + self.rk4_step(z, H, direction)
            ttt = ttt + H_flat
            Z[i + 1, :, :] += z
            T[i + 1, :] += direction * ttt
        return T, Z, z

    def rk4_step(self, z, H, sign=1.0):
        """One RK4 solver step."""
        k1 = sign * self.f(0.0, z)
        k2 = sign * self.f(0.0, z + 0.5 * H * k1)
        k3 = sign * self.f(0.0, z + 0.5 * H * k2)
        k4 = sign * self.f(0.0, z + 1.0 * H * k3)
        delta = H / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return delta

    @torch.no_grad()
    def solve_numpy(self, z0: np.ndarray, t, direction, n_steps: int = 30):
        """Solve trajectories and give result as numpy arrays."""
        z0t = torch.from_numpy(z0).float()
        tt = torch.from_numpy(t).float()
        T_TRAJ, Z_TRAJ, _ = self.solve(z0t, tt, n_steps, direction)
        Z_TRAJ = Z_TRAJ.detach().cpu().numpy()
        T_TRAJ = T_TRAJ.detach().cpu().numpy()
        return T_TRAJ, Z_TRAJ

    @torch.no_grad()
    def f_numpy(self, z):
        """Evaluate vector field using numpy arrays."""
        zt = torch.from_numpy(z).float()
        f = self.f(0.0, zt)
        f = f.detach().cpu().numpy()
        return f


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

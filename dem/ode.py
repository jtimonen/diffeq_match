import torch
import torch.nn as nn
import numpy as np
from .networks import TanhNetTwoLayer


class ODE(nn.Module):
    """An ordinary differential equation module."""

    def __init__(self, D: int, H: int = 32):
        super().__init__()
        self.f = TanhNetTwoLayer(D, D, n_hidden=H)
        self.D = D

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

import torch
import torch.nn as nn
import numpy as np
from dem.modules.vectorfield import VectorField, StochasticVectorField
from dem.modules.networks import Reverser
from dem.utils.utils import tensor_to_numpy


class GenModel(nn.Module):
    """Main model module."""

    def __init__(self, net_f: nn.Module, stochastic: bool = False):
        super().__init__()
        self.stochastic = stochastic
        if stochastic:
            self.field = StochasticVectorField(net_f)
        else:
            self.field = VectorField(net_f)
        self.field_reverse = VectorField(Reverser(net_f))

    def _traj_forward(self, y_init: torch.Tensor, ts, sde: bool, **kwargs):
        if sde:
            if not self.is_stochastic:
                raise RuntimeError("vector field is not stochastic!")
            return self.field.sdeint(y_init, ts, **kwargs)
        else:
            return self.field.odeint(y_init, ts, **kwargs)

    def _traj_backward(self, y_init: torch.Tensor, ts, **kwargs):
        return self.field_reverse.odeint(y_init, ts, **kwargs)

    def traj(
        self, y_init: torch.Tensor, ts, sde=False, backward: bool = False, **kwargs
    ):
        """Compute ODE or SDE trajectories

        :return: a torch tensor with shape (n_trajectories, n_points, n_dims)
        """
        if not backward:
            out = self._traj_forward(y_init, ts, sde=sde, **kwargs)
        else:
            if sde:
                raise RuntimeError("cannot integrate sde backwards!")
            out = self._traj_backward(y_init, ts, **kwargs)
        return out.permute(1, 0, 2)

    @torch.no_grad()
    def traj_numpy(
        self,
        y_init: np.ndarray,
        ts: np.ndarray,
        sde=False,
        backward: bool = False,
        **kwargs
    ):
        y_init = torch.from_numpy(y_init).float()
        ts = torch.from_numpy(ts).float()
        y_traj = self.traj(y_init, ts, sde, backward, **kwargs)
        return tensor_to_numpy(y_traj)

    def forward(self, y_init: torch.Tensor, N: int = 60):
        ts = torch.linspace(0, 1, N).float()
        y_traj = self.traj(y_init, ts, sde=True, forward=True)
        return torch.transpose(y_traj.diagonal(), 0, 1)

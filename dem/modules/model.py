import torch
import torch.nn as nn
import numpy as np
from dem.modules.vectorfield import VectorField, StochasticVectorField
from dem.modules.networks import Reverser
from dem.utils.utils import tensor_to_numpy, add_noise
from dem.data.dataset import NumpyDataset
from dem.plotting.model import plot_model_state


class Stage:
    """Stage of a generative model."""

    def __init__(
        self,
        sde: bool = False,
        backwards: bool = False,
        t_max: float = 1.0,
        uniform: bool = True,
        sigma: float = 0.0,
    ):
        if sde and backwards:
            raise ValueError("Cannot create a Stage with sde=True and backwards=True")
        assert t_max > 0.0, "t_max must be positive!"
        self.sde = sde
        self.backwards = backwards
        self.uniform = uniform
        self.t_max = t_max
        self.sigma = sigma

    def description(self):
        str0 = str(round(self.sigma, 3))
        str1 = "SDE" if self.sde else "ODE"
        str2 = "backwards" if self.backwards else "forward"
        str3 = str(round(self.t_max, 3))
        if self.sigma > 0.0:
            desc = (
                "Add Gaussian noise with std" + str0 + "to initial "
                "values of the stage."
            )
        else:
            desc = ""
        desc += "Integrate " + str1 + " " + str2 + " for time " + str3 + "."
        if self.uniform:
            desc += " Return outputs at times [0, ..., " + str3 + "] uniformly."
        return desc

    def __repr__(self):
        return "<Stage: " + self.description() + ">"


class PriorInfo:
    """The prior information."""

    def __init__(self, init: np.ndarray):
        super().__init__()
        self.init_data = NumpyDataset(init)

    def draw(self, N: int, replace: bool = True):
        N_prior = len(self.init_data)
        inds = np.random.choice(N_prior, N, replace=replace)
        return self.init_data[inds]


class DynamicModel(nn.Module):
    """Dynamic part of the generative model."""

    def __init__(self, net_f: nn.Module, stochastic: bool = False):
        super().__init__()
        self.stochastic = stochastic
        if stochastic:
            self.field = StochasticVectorField(net_f)
        else:
            self.field = VectorField(net_f)
        self.field_reverse = VectorField(Reverser(net_f))

    def description(self):
        str0 = str(self.stochastic)
        str1 = str(type(self.field.net_f).__name__)
        desc = "DynamicModel(stochastic=" + str0 + ", net_f=" + str1 + ")"
        return desc

    def __repr__(self):
        return "<" + self.description() + ">"

    def _traj_forward(self, y_init: torch.Tensor, ts, sde: bool, **kwargs):
        if sde:
            if not self.stochastic:
                raise RuntimeError("vector field is not stochastic!")
            return self.field.sdeint(y_init, ts, **kwargs)
        else:
            return self.field.odeint(y_init, ts, **kwargs)

    def _traj_backward(self, y_init: torch.Tensor, ts, **kwargs):
        return self.field_reverse.odeint(y_init, ts, **kwargs)

    def traj(
        self,
        y_init: torch.Tensor,
        ts: torch.Tensor,
        sde=False,
        backwards: bool = False,
        **kwargs
    ):
        """Compute ODE or SDE trajectories

        :return: a torch tensor with shape (n_trajectories, n_points, n_dims)
        """
        if not backwards:
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

    def traj_linspace(self, y_init: torch.Tensor, L: int, t_max: float, **kwargs):
        ts = torch.linspace(0.0, t_max, L).type_as(y_init)
        return self.traj(y_init=y_init, ts=ts, **kwargs)

    def traj_diag(self, y_init: torch.Tensor, L: int, t_max: float, **kwargs):
        y_traj = self.traj_linspace(y_init, L, t_max, **kwargs)
        return torch.transpose(y_traj.diagonal(), 0, 1)

    def forward(self, y_init: torch.Tensor, t_max: float, **kwargs):
        ts = torch.tensor([0.0, t_max]).type_as(y_init)
        y_traj = self.traj(y_init=y_init, ts=ts, **kwargs)
        return y_traj[:, 1, :]


class GenerativeModel(nn.Module):
    """The generative model."""

    def __init__(
        self,
        dynamics: DynamicModel,
        prior_info: PriorInfo,
        stages=None,
        solver_kwargs=None,
    ):
        super().__init__()
        self.dynamics = dynamics
        self.prior_info = prior_info
        if stages is None:
            stages = [Stage()]
        self.stages = stages
        if solver_kwargs is None:
            solver_kwargs = dict()
        self.solver_kwargs = solver_kwargs

    @property
    def num_stages(self):
        return len(self.stages)

    def __repr__(self):
        desc = "GenerativeModel with " + self.dynamics.description() + ". Stages:\n"
        for s in self.stages:
            desc += " - " + s.description() + "\n"
        return desc

    def _generate_init(self, N: int, like=None):
        """Perform initial stage of the generative process.

        :param N: number of points to generate from the process
        :param like: A torch.Tensor that defines the type and device of used tensors.
        """
        if like is None:
            like = torch.tensor([0.0], dtype=torch.float32)
        init = self.prior_info.draw(N, replace=True)
        return torch.from_numpy(init).type_as(like)

    def _perform_stage(self, x: torch.Tensor, stage: Stage, traj=False, **kwargs):
        """Perform one dynamic stage of the generative process.

        :param x: State after previous stage, tensor of shape (N, D).
        :param stage: Description of the stage.
        :param kwargs: Keyword arguments to the ODE or SDE solver.
        :param traj: should solve also a dense trajectory (should be False during
        training and used only for visualization)
        :return: A tensor of shape (N, D)
        """
        x = add_noise(x, stage.sigma)
        N = x.shape[0]
        rev = stage.backwards
        sde = stage.sde
        if stage.uniform:
            x_new = self.dynamics.traj_diag(
                x, L=N, sde=sde, backwards=rev, t_max=stage.t_max, **kwargs
            )
        else:
            x_new = self.dynamics(
                x, sde=sde, backwards=rev, t_max=stage.t_max, **kwargs
            )
        if traj:
            x_traj = self.dynamics.traj_linspace(x, 30, stage.t_max, **kwargs)
        else:
            x_traj = None
        return x_new, x_traj

    def forward(self, N: int, like=None):
        """Forward pass through the generative model."""
        x = self._generate_init(N=N, like=like)
        for s in self.stages:
            x, _ = self._perform_stage(x, s, False, **self.solver_kwargs)
        return x

    @torch.no_grad()
    def forward_numpy(self, N: int):
        x = self._generate_init(N=N, like=None)
        x_all = [x]
        traj_all = []
        for s in self.stages:
            x, traj = self._perform_stage(x, s, True, **self.solver_kwargs)
            x_all += [x]
            traj_all += [traj]
        x_all = [tensor_to_numpy(x) for x in x_all]
        traj_all = [tensor_to_numpy(traj) for traj in traj_all]
        return x_all, traj_all

    @torch.no_grad()
    def visualize(self, N: int, data=None, save_name=None, save_dir=None, **kwargs):
        x_all, traj_all = self.forward_numpy(N)
        plot_model_state(x_all, traj_all, data, save_name, save_dir, **kwargs)

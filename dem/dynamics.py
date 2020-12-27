import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from scdyn.model.networks import TimeEncoder
from scdyn.utils.logging import log_info
from scdyn.inference.training_dynamics import train_dynamics
from scdyn.model.ode import create_ode
from matplotlib import pyplot as plt
from scdyn.utils.math import MMD


def create_dynamics(
    D: int,
    z0_mean: np.ndarray,
    z0_s2: np.ndarray,
    H: int = 64,
    H_ode: int = 64,
    name: str = "dynamics",
):
    """A convenience function for creating dynamics."""
    nn_t = TimeEncoder(D, H)
    ode = create_ode(D, H_ode)
    model = LatentDynamics(nn_t, ode, z0_mean, z0_s2, name)
    model.eval()
    return model


class LatentDynamics(nn.Module):
    """Model for latent dynamics."""

    def __init__(
        self, nn_t, ode, z0_mean: np.ndarray, z0_s2: np.ndarray, name: str = "dynamics"
    ):
        super().__init__()
        self.D = nn_t.D
        self.enc_t = nn_t
        self.ode = ode
        self.n_steps = 30
        self.name = name
        self.file = self.name + ".pth"
        self.history = None
        self.z0_mean = z0_mean
        self.z0_s2 = z0_s2
        self.mmd = MMD(self.D, ell2=0.05)

    def forward(self, z0: torch.Tensor, t: torch.Tensor):
        """Forward pass."""
        # Position to latent time
        # t = self.enc_t(z)

        # Dynamics
        S = self.n_steps
        _, _, zf = self.ode.solve(z0, t, S, 1.0)

        return zf

    @torch.no_grad()
    def forward_numpy(self, z0, t):
        """Forward pass but using numpy arrays."""
        t = torch.from_numpy(t).float()
        z0 = torch.from_numpy(z0).float()
        zf = self.forward(z0, t)
        zf = zf.detach().cpu().numpy()
        return zf

    @torch.no_grad()
    def draw_z0(self, n: int, rng=np.random.default_rng()):
        """Take n draws from p(z0)."""
        loc = np.tile(self.z0_mean, (n, 1))
        scale = np.tile(self.z0_s2, (n, 1))
        return rng.normal(loc=loc, scale=scale)

    @torch.no_grad()
    def draw_t(self, n: int, rng=np.random.default_rng()):
        """Take n draws from p(t)."""
        return rng.uniform(size=(n, 1))

    def full_inference(self, dataset, save_name=None):
        """Full data set inference after parameters have been trained."""
        z = dataset.z.T
        n = z.shape[0]
        z0 = self.draw_z0(n)
        t = self.draw_t(n)
        zt = self.forward_numpy(z0, t)
        t = t.flatten()
        loss = self.loss_numpy(z0, z, zt, t)

        # First plot
        plt.scatter(z[:, 0], z[:, 1], c=t)
        plt.colorbar()
        plt.title("mean loss = " + str(np.mean(loss)))
        plt.show()

        # Second plot
        plt.scatter(z[:, 0], z[:, 1], alpha=0.3)
        plt.scatter(zt[:, 0], zt[:, 1], alpha=0.3)
        f = self.ode.f_numpy(z)
        plt.quiver(z[:, 0], z[:, 1], f[:, 0], f[:, 1])
        plt.title("mean loss = " + str(np.mean(loss)))
        plt.show()

    def loss(self, z0, z, zt, t):
        """Compute loss."""
        t0 = self.enc_t(z0)
        loss0 = t0 ** 2
        loss1 = self.mmd(z, zt)
        return loss0 + loss1

    @torch.no_grad()
    def loss_numpy(self, z0, z, zt, t):
        """Compute loss but with numpy arrays."""
        z0 = torch.from_numpy(z0).float()
        z = torch.from_numpy(z).float()
        zt = torch.from_numpy(zt).float()
        t = torch.from_numpy(t).float()
        loss = self.loss(z0, z, zt, t)
        return loss.detach().cpu().numpy()

    @torch.no_grad()
    def load_state(self, file, parent_dir="."):
        """Load model state dict from file.

        :raises: FileNotFoundError
        """
        load_path = os.path.join(parent_dir, file)
        if os.path.isfile(load_path):
            log_info("Loading model state dict from: " + load_path)
            state_dict = torch.load(load_path)
            self.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(
                "Unable to load state. File not found: " + load_path
            )

    @torch.no_grad()
    def save_state(self, file="dyn.pth", parent_dir="."):
        """Save model state dict to file."""
        save_path = os.path.join(parent_dir, file)
        log_info("Saving model state dict to: " + save_path)
        torch.save(obj=self.state_dict(), f=save_path)

    @torch.no_grad()
    def info(self):
        """Prints info about the model."""
        print("A LatentDynamics model:")
        print(" - latent dimension = " + str(self.D))

    def fit(
        self,
        dataset,
        p_valid: float = 0.25,
        batch_size: int = 128,
        seed: int = 123,
        lr: float = 1e-3,
        epochs: int = 100,
        print_frequency: int = 1,
        save_frequency=0,
        save_name: str = None,
        debug: bool = False,
    ):
        """Train the ODE parameters."""
        wd = 1e-6
        adam_eps = 0.01
        mode = 2
        params_list = list(self.parameters())
        optim = Adam(params_list, eps=adam_eps, lr=lr, weight_decay=wd)
        history = train_dynamics(
            self,
            dataset,
            optim,
            mode,
            epochs,
            print_frequency,
            save_frequency,
            save_name,
            p_valid,
            batch_size,
            seed,
            debug,
        )
        self.history = history
        if save_name is not None:
            fn = save_name + ".pth"
            self.save_state(fn)

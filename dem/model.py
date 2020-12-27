import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from scdyn.model.networks import TimeEncoder
from scdyn.utils.logging import log_info
from scdyn.utils.utils import normalize_torch
from scdyn.inference.training import train_model
from scdyn.model.ode import create_ode
from scdyn.model.vae import create_vae
from scdyn.utils.math import OdeTarget
from scdyn.utils.utils import dot_torch


def create_model(
    G: int, D: int, H_enc=128, H_dec=0, H_ode=16, name: str = "scdyn_model"
):
    """A convenience function for creating a *ScdynModel* instance with default
    network structures and options.

    :return: a model instance
    :rtype: ScdynModel
    """
    vae = create_vae(G, D, H_enc, H_dec)
    enc_t = TimeEncoder(D, 64)
    ode = create_ode(vae.D, H_ode)
    model = ScdynModel(vae, enc_t, ode, name)
    model.eval()
    return model


class ScdynModel(nn.Module):
    """VAE for expression levels."""

    def __init__(self, vae, enc_t, ode, name: str = "scdyn_model"):
        super().__init__()
        self.G = vae.G
        self.D = vae.D
        self.k = 16
        self.vae = vae
        self.enc_t = enc_t
        self.ode = ode
        self.n_steps = 30
        self.name = name
        self.file = self.name + ".pth"
        self.history = None
        self.z0 = None
        self.ode_target = OdeTarget(D=self.D, N=self.k - 1)

    def forward(
        self,
        xu: torch.Tensor,
        xs: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
        f: torch.Tensor,
        z0: torch.Tensor,
        mode: int,
    ):
        """Forward pass."""

        if mode == 0:

            # VAE
            xs = normalize_torch(xs)
            kl, rec_loss = self.vae(xs)
            loss_terms = torch.cat((kl.view(1), rec_loss.view(1), torch.zeros(3)))

        elif mode == 1:

            # Field init
            v = self.ode.f(0.0, z)
            abs_f = torch.norm(f, p=2, dim=1, keepdim=True)
            abs_v = torch.norm(v, p=2, dim=1, keepdim=True)
            cos_loss = -dot_torch(f, v) / (abs_f * abs_v)
            cos_loss = cos_loss.mean()
            loss_terms = 1000 * torch.cat((torch.zeros(4).float(), cos_loss.view(1)))

        elif mode == 2:

            # Dynamics
            S = self.n_steps
            _, _, zb = self.ode.solve(z, t, S, -1.0)
            v = self.ode.f(0.0, z)
            abs_f = torch.norm(f, p=2, dim=1, keepdim=True)
            abs_v = torch.norm(v, p=2, dim=1, keepdim=True)
            cos_loss = -dot_torch(f, v) / (abs_f * abs_v)
            cos_loss = cos_loss.mean()
            zero_loss = 0.1 * torch.sqrt((zb - z0.view(1, -1)).pow(2).sum(dim=1)).mean()
            loss_terms = 1000 * torch.cat(
                (torch.zeros(3).float(), cos_loss.view(1), zero_loss.view(1))
            )

        else:
            raise ValueError("invalid mode!")

        return loss_terms

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
    def save_state(self, file="model.pth", parent_dir="."):
        """Save model state dict to file."""
        save_path = os.path.join(parent_dir, file)
        log_info("Saving model state dict to: " + save_path)
        torch.save(obj=self.state_dict(), f=save_path)

    @torch.no_grad()
    def info(self):
        """Prints info about the model."""
        print("ScdynModel:")
        print(" - number of genes = " + str(self.G))
        print(" - latent dimension = " + str(self.D))

    @torch.no_grad()
    def ode_init(self, vae_fit):
        """Initialize model vector field according to VAE fit."""
        log_info("Initializing model vector field according to VAE fit.")
        u = vae_fit.mst.u0
        v = vae_fit.mst.v0
        ell = 0.5 * np.mean(vae_fit.mst.edge_lengths)
        self.ode_target.initialize(u, v, ell)

    def fit_vae(
        self,
        dataset,
        p_valid: float = 0.25,
        batch_size: int = 128,
        seed: int = 123,
        lr: float = 1e-3,
        epochs: int = 100,
        print_frequency: int = 1,
        save_frequency=0,
        save_name=None,
        debug: bool = False,
    ):
        """Train the VAE parameters."""
        wd = 1e-6
        adam_eps = 0.01
        mode = 0
        params_list = list(self.vae.parameters())
        optim = Adam(params_list, eps=adam_eps, lr=lr, weight_decay=wd)
        history = train_model(
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
        dataset.add_vae_fit(self)
        self.ode_init(dataset.vae_fit)
        if save_name is not None:
            fn = save_name + ".pth"
            self.save_state(fn)

    def fit_fld(
        self,
        dataset,
        p_valid: float = 0.25,
        batch_size: int = 128,
        seed: int = 123,
        lr: float = 1e-3,
        epochs: int = 100,
        print_frequency: int = 1,
        save_frequency=0,
        save_name: str = "model",
        debug: bool = False,
    ):
        """Train the ODE parameters."""
        wd = 1e-6
        adam_eps = 0.01
        mode = 1
        params_list = list(self.ode.parameters())
        optim = Adam(params_list, eps=adam_eps, lr=lr, weight_decay=wd)
        history = train_model(
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

    def fit_dyn(
        self,
        dataset,
        p_valid: float = 0.25,
        batch_size: int = 128,
        seed: int = 123,
        lr: float = 1e-3,
        epochs: int = 100,
        print_frequency: int = 1,
        save_frequency=0,
        save_name: str = "model",
        debug: bool = False,
    ):
        """Train the ODE parameters."""
        wd = 1e-6
        adam_eps = 0.01
        mode = 2
        params_list = list(self.ode.parameters())
        optim = Adam(params_list, eps=adam_eps, lr=lr, weight_decay=wd)
        history = train_model(
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

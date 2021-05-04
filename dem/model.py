import os
import torch
import torch.nn as nn
import torchsde
from pytorch_lightning import Trainer
import pytorch_lightning as pl


from .vectorfield import VectorField, Reverser
from .priorinfo import PriorInfo, create_prior_info
from .kde import KdeDiscriminator
from .data import create_dataloader, MyDataset
from .callbacks import MyCallback
from pytorch_lightning.callbacks import ModelCheckpoint


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

    def fit(
        self,
        z_data,
        batch_size=128,
        n_epochs: int = 400,
        lr: float = 0.005,
        plot_freq=0,
    ):
        self.set_bandwidth(z_data)
        z_data = torch.from_numpy(z_data).float()
        learner = TrainingSetup(
            self,
            z_data,
            batch_size,
            lr,
            plot_freq,
        )
        save_path = learner.outdir

        checkpoint_callback = ModelCheckpoint(
            verbose=True, monitor="valid_loss", mode="min", prefix="mod"
        )

        trainer = Trainer(
            min_epochs=n_epochs,
            max_epochs=n_epochs,
            default_root_dir=save_path,
            callbacks=[MyCallback(), checkpoint_callback],
        )
        trainer.fit(learner)


class TrainingSetup(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        z_data,
        batch_size: int,
        lr_init: float,
        plot_freq=0,
        outdir="out",
    ):
        super().__init__()
        num_workers = 0
        ds = MyDataset(z_data)
        train_loader = create_dataloader(ds, batch_size, num_workers, shuffle=True)
        valid_loader = create_dataloader(ds, None, num_workers, shuffle=False)
        self.N = len(train_loader.dataset)
        self.model = model
        # self.disc = disc
        # if disc.D != self.model.D:
        #    raise RuntimeError("Discriminator dimension incompatible with model!")
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lr_init = lr_init
        self.plot_freq = plot_freq

        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        self.outdir = outdir

    def forward(self, n_draws: int):
        return self.model(n_draws)

    # KDE Stuff
    def kde_terms(self, z_data, z_samples):
        loss1 = -torch.mean(self.model.kde(z_samples, z_data))
        loss2 = -torch.mean(self.model.kde(z_data, z_samples))
        return loss1, loss2

    # GAN STUFF
    # def loss_generator(self, z_samples):
    #    """Rather than training G to minimize log(1 − D(G(z))), we can train G to
    #    maximize log D(G(z)). This objective function results in the same fixed point
    #    of the dynamics of G and D but provides much stronger gradients early in
    #    learning. (Goodfellow et al., 2014)
    #    """
    #    G_z = z_samples
    #    D_G_z = self.disc(G_z)  # classify fake data
    #    loss_fake = -torch.mean(log_eps(D_G_z))
    #    return loss_fake
    #
    # def loss_discriminator(self, z_samples, z_data):
    #    """Discriminator loss."""
    #    D_x = self.disc(z_data)  # classify real data
    #    G_z = z_samples
    #    D_G_z = self.disc(G_z.detach())  # classify fake data
    #   loss_real = -torch.mean(log_eps(D_x))
    #   loss_fake = -torch.mean(log_eps(1 - D_G_z))
    #   loss = 0.5 * (loss_real + loss_fake)
    #   return loss

    def training_step(self, data_batch, batch_idx):
        z_data = data_batch
        N = z_data.size(0)
        z_fake = self.model(N)  # generate fake data
        lt1, lt2 = self.kde_terms(z_data, z_fake)
        return 0.5 * (lt1 + lt2)

        # GAN STUFF
        # if optimizer_idx == 0:
        #    loss = self.loss_generator(z_fake)
        # elif optimizer_idx == 1:
        #    loss = self.loss_discriminator(z_fake, z_data)
        # else:
        #    raise RuntimeError("optimizer_idx must be 0 or 1!")
        # return loss

    def validation_step(self, data_batch, batch_idx):
        z_data = data_batch
        N = z_data.size(0)
        z_fake = self.model(N)  # generate fake data
        lt1, lt2 = self.kde_terms(z_data, z_fake)
        loss = 0.5 * (lt1 + lt2)
        self.log("valid_loss", loss)
        idx_epoch = self.current_epoch
        pf = self.plot_freq
        if pf > 0:
            if idx_epoch % pf == 0:
                self.visualize(z_fake, z_data, loss, idx_epoch)
                self.sde_viz(z_data, idx_epoch)
        return loss

    @torch.no_grad()
    def check_model(self, z_data):
        z_data = torch.from_numpy(z_data).float()
        N = z_data.size(0)
        z_fake = self.model(N)  # generate fake data
        lt1, lt2 = self.kde_terms(z_data, z_fake)
        loss = 0.5 * (lt1 + lt2)
        print("valid_loss", loss)
        idx_epoch = -1
        self.visualize(z_fake, z_data, loss, idx_epoch)
        self.sde_viz(z_data, idx_epoch)

    @torch.no_grad()
    def generate_traj(self, N: int = 30, z_init=None):
        """Returns tensor of shape (L, N, D)."""
        L = 100
        ts = torch.linspace(0, 1, L).float()
        if z_init is None:
            z_init = self.model.draw_init(N)
        z_traj = self.model.traj(z_init, ts, sde=True, forward=True)
        return z_traj

    @torch.no_grad()
    def visualize(self, z_samp, z_data, loss, idx_epoch):
        fig_dir = os.path.join(self.outdir, "figs")
        if not os.path.isdir(fig_dir):
            os.mkdir(fig_dir)
        z_samp = z_samp.detach().cpu().numpy()
        z_data = z_data.detach().cpu().numpy()
        if self.model.D == 2:
            plot_state_2d(self.model, z_samp, z_data, idx_epoch, loss, fig_dir)
        elif self.model.D == 3:
            plot_state_3d(self.model, z_samp, z_data, idx_epoch, loss, fig_dir)
        else:
            plot_state_nd(self.model, z_samp, z_data, idx_epoch, loss, None, fig_dir)

    @torch.no_grad()
    def sde_viz(self, z_data, idx_epoch):
        print(" ")
        print("kde_sigma =", self.model.kde.sigma)
        print("diffusion_magnitude =", self.model.field.diffusion_magnitude)
        N_TRAJ = 30  # number of trajectories
        L_TRAJ = 100  # number of points per trajectory
        fig_dir = os.path.join(self.outdir, "figs")
        z_init = self.model.draw_init(N_TRAJ)
        ts = torch.linspace(0, 1, L_TRAJ).float()
        z_traj = torchsde.sdeint(self.model.field, z_init, ts, method="euler")
        z_traj = z_traj.detach().cpu().numpy()
        z_data = z_data.detach().cpu().numpy()
        if self.model.D == 2:
            plot_sde_2d(self.model, z_data, z_traj, idx_epoch, save_dir=fig_dir)
        elif self.model.D == 3:
            plot_sde_3d(self.model, z_data, z_traj, idx_epoch, save_dir=fig_dir)
        else:
            plot_sde_nd(self.model, z_data, z_traj, idx_epoch, save_dir=fig_dir)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.model.parameters(), lr=self.lr_init)
        # opt_d = torch.optim.Adam(self.disc.parameters(), lr=self.lr_init)
        return opt_g

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

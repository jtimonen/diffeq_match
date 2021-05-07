import os
import abc
import pytorch_lightning as pl

from dem.data.dataloader import create_dataloader


class Learner(pl.LightningModule, abc.ABC):
    def __init__(
        self,
        train_dataset,
        valid_dataset,
        train_batch_size: int,
        valid_batch_size=None,
        plot_freq=0,
        outdir="out",
        num_workers: int = 0,
    ):
        super().__init__()
        train_loader = create_dataloader(
            train_dataset, train_batch_size, num_workers, shuffle=True
        )
        valid_loader = create_dataloader(
            valid_dataset, valid_batch_size, num_workers, shuffle=False
        )
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.plot_freq = plot_freq
        self.outdir = None
        self.set_outdir(outdir)

    @property
    def num_train(self):
        return len(self.train_loader.dataset)

    @property
    def num_valid(self):
        return len(self.valid_loader.dataset)

    def set_outdir(self, path):
        """Set output directory."""
        if not os.path.isdir(path):
            os.mkdir(path)
        else:
            raise RuntimeError("Output directory (" + path + ") exists!")
        self.outdir = dir

    def validation_epoch_end(self, outputs) -> None:
        idx_epoch = self.current_epoch
        pf = self.plot_freq
        if pf > 0:
            if idx_epoch % pf == 0:
                self.visualize()
        return None

    @abc.abstractmethod
    def visualize(self, *args):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

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

    # GAN STUFF
    # if optimizer_idx == 0:
    #    loss = self.loss_generator(z_fake)
    # elif optimizer_idx == 1:
    #    loss = self.loss_discriminator(z_fake, z_data)
    # else:
    #    raise RuntimeError("optimizer_idx must be 0 or 1!")
    # return loss

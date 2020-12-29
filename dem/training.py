import torch
import torch.nn as nn
import pytorch_lightning as pl


class Learner(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        dataloader,
        mmd: nn.Module,
        lr: float = 0.005,
        n_draws=None,
    ):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.lr = lr
        self.mmd = mmd
        if n_draws is None:
            n_draws = len(dataloader.dataset)
        self.n_draws = n_draws

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        z_data = batch
        z0, t, z = self.model(self.n_draws)
        loss = self.mmd(z_data, z)
        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_dataloader(self):
        return self.dataloader


def train_model(model, dataloader, optim, n_draws: int, n_epochs: int):
    model.train()
    S = 30
    for j in range(0, n_epochs):
        for idx, z_batch in enumerate(dataloader):
            optim.zero_grad()
            z0, t = model.draw_z0t0(N=n_draws)
            zf = model(z0, t, n_steps=S)
            loss = model.loss(z_batch, zf)
            print("epoch = " + str(j) + ", loss = " + str(loss))
            loss.backward()
            optim.step()
            if j % (n_epochs / 10) == 0:
                _, z_traj, _ = model.ode.solve(z0, t, n_steps=S)
                z_traj = z_traj.detach().cpu().numpy()
                model.plot_forward(z_batch, zf, z_traj)

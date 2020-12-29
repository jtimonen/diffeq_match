import torch
import numpy as np


def train_model(model, dataloader, optim, n_draws: int, n_epochs: int):
    model.train()
    for j in range(0, n_epochs):
        optim.zero_grad()
        for idx, z_batch in enumerate(dataloader):
            z0, t = model.draw_z0t0(N=n_draws)
            zf = model(z0, t, n_steps=30)
            loss = model.loss(z_batch, zf)
            print("epoch = " + str(j) + ", loss = " + str(loss))
            loss.backward()
            optim.step()
            if j % (n_epochs / 10) == 0:
                model.plot_forward(z_batch, zf)

import torch
import numpy as np


def train_model(model, z_data, optim, n_epochs: int):
    model.train()
    N = z_data.shape[0]
    for j in range(0, n_epochs):
        optim.zero_grad()
        z0, t = model.draw_z0t0(N=N)
        zf = model(z0, t, n_steps=30)
        loss = model.loss(z_data, zf)
        print("epoch = " + str(j) + ", loss = " + str(loss))
        loss.backward()
        optim.step()

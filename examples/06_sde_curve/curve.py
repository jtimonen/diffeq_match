from dem import sim, plot_sim
from dem import GenModel

import torch

N = 1000
z_data, t_data = sim(1, N, sigma=0.1)
# plot_sim(z_data, t_data)

i_loc = [1.5, 0.2]
i_std = [0.07]

model = GenModel(i_loc, i_std, n_hidden=64)
print(model)

zz = torch.from_numpy(z_data).float()

# Create and fit discriminator
# disc = Discriminator(D=model.D)
# disc.fit(zz, plot_freq=50, n_epochs=300, lr=0.005)

# Create and fit model
model.fit(zz, plot_freq=5, n_epochs=500, lr=0.005, batch_size=250)

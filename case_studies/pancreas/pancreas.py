import scanpy as sc

# fp = "/Users/juhotimonen/Work/research/SC/single-cell-data/pancreas/data/Pancreas/endocrinogenesis_day15.h5ad"
# adata = sc.read(fp)

from dem import GenModel
import torch
import numpy as np

z_data = np.loadtxt("precomp/txt/latent.txt")

i_loc = [[-2.0, 0.0, -1.5]]
i_std = [0.1]

model = GenModel(i_loc, i_std, n_hidden=48, sigma=0.2)
zz = torch.from_numpy(z_data).float()

# Create and fit model
model.fit(zz, plot_freq=5, n_epochs=500, lr=0.005, batch_size=256)

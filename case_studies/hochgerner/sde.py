import scanpy as sc
#fp = "/Users/juhotimonen/Work/research/SC/single-cell-data/pancreas/data/Pancreas/endocrinogenesis_day15.h5ad"
#adata = sc.read(fp)

from dem import GenModel
import torch
import numpy as np
import os

model_dir = "lin5_D6_G1200_H128_MC20-dentate-gyrus-neurogenesis_hochgerner"
z_data = np.loadtxt(os.path.join(model_dir, "latent.txt"))
start_idx = int(np.loadtxt(os.path.join(model_dir, "start_idx.txt")))


i_loc = np.array(z_data[start_idx, :]).reshape((1, -1))
i_std = [0.1]
print("i_loc:", i_loc)

model = GenModel(i_loc, i_std, n_hidden=64, sigma=0.2)
zz = torch.from_numpy(z_data).float()

# Create and fit model
model.fit(zz, plot_freq=5, n_epochs=500, lr=0.005, batch_size=500)




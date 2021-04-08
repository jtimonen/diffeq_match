import torch

from dem import load_data_txt, GenModel
from dem.model import TrainingSetup

# Load data
pdir = "nl19_D3_G2000_H64_MC30-real--silver--dentate-gyrus-neurogenesis_hochgerner"
z_data, z0, _, _ = load_data_txt(pdir)

# Create and fit model
model = GenModel(z0, n_hidden=64, azimuth_3d=-79, elevation_3d=52, H_3d=2.75)
PATH = "out/lightning_logs/version_0/checkpoints/mod-epoch=52-step=794.ckpt"
ts = TrainingSetup.load_from_checkpoint(
    PATH,
    model=model,
    z_data=z_data,
    plot_freq=1,
    n_epochs=100,
    lr_init=0.005,
    batch_size=250,
)

ts.check_model(z_data)

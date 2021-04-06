from dem import load_data_txt, GenModel

# Load data
pdir = "nl19_D2_G800_H32_MC20-real--silver--dentate-gyrus-neurogenesis_hochgerner"
z_data, z0, _, _ = load_data_txt(pdir)

# Create and fit model
model = GenModel(z0, n_hidden=64)
model.fit(z_data, plot_freq=5, n_epochs=100, lr=0.005, batch_size=250)

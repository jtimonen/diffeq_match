from dem import load_data_txt, GenModel

# Load data
pdir = "lin19_D2_G1200_H128_MC20-real--silver--dentate-gyrus-neurogenesis_hochgerner"
z_data, z0, _, _ = load_data_txt(pdir)

# Create and fit model
model = GenModel(z0, n_hidden=32)
model.fit(z_data, plot_freq=5, n_epochs=100, lr=0.005, batch_size=250)

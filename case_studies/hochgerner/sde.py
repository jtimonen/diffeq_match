from dem import load_data_txt, Discriminator, GenModel

# Load data
pdir = "nl19_D2_G2000_H128_MC20-real--silver--dentate-gyrus-neurogenesis_hochgerner"
z_data, z0, _, _ = load_data_txt(pdir)

# Create and fit discriminator
disc = Discriminator(D=2)
disc.fit(z_data, lr=0.005, plot_freq=20, n_epochs=400)

# model = GenModel(z0, n_hidden=64, sigma=0.1)

# Create and fit model
# model.fit(z_data, plot_freq=5, n_epochs=60, lr=0.005, batch_size=250)

from dem import load_data_txt, Discriminator, GenModel

# Load data
pdir = "nl19_D2_G800_H32_MC20-real--silver--dentate-gyrus-neurogenesis_hochgerner"
z_data, z0, _, _ = load_data_txt(pdir)

# Create and fit discriminator
disc = Discriminator(D=2)
disc.fit(z_data, lr=0.005, n_epochs=100, plot_freq=25)

model = GenModel(z0, n_hidden=64)

# Create and fit model
model.fit(z_data, disc, plot_freq=5, n_epochs=100, lr=0.005, batch_size=250)

from dem import load_data_txt, GenModel

# Load data
pdir = "nl19_D3_G2000_H64_MC30-real--silver--dentate-gyrus-neurogenesis_hochgerner"
z_data, z0, _, _ = load_data_txt(pdir)

# Create and fit model
model = GenModel(z0, n_hidden=64, azimuth_3d=-79, elevation_3d=52)
model.fit(z_data, plot_freq=1, n_epochs=100, lr=0.005, batch_size=250)

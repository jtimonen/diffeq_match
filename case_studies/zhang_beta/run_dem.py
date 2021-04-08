from dem import load_data_txt, GenModel

# Load data
pdir = "lin12_D2_G1000_H64_MC20-real--gold--pancreatic-beta-cell-maturation_zhang/"
z_data, z0, _, _ = load_data_txt(pdir)

# Create and fit model
model = GenModel(z0)
model.fit(z_data, plot_freq=1, n_epochs=100, lr=0.005, batch_size=128)

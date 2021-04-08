from dem import load_data_txt, GenModel

# Load data
pdir = "lin26_D2_G1200_H64_MC30-real--silver--fibroblast-reprogramming_treutlein"
z_data, z0, _, _ = load_data_txt(pdir)
print(z_data.shape)

# Create and fit model
model = GenModel(z0)
model.fit(z_data, plot_freq=1, n_epochs=100, lr=0.005, batch_size=128)

import dem

# Simulate data
N = 1000
z_data, t_data = dem.sim(1, N, sigma=0.1)
# dem.plot_sim(z_data, t_data)
z_init = z_data[0:30, :]

# Create model and discriminator
model = dem.create_model(init=z_init)
disc = dem.create_discriminator(D=model.D, fixed_kde=True)

# Training
dem.train_model(model, disc, z_data, plot_freq=3, n_epochs=6, lr=0.002, batch_size=128)

from pytorch_lightning.callbacks import Callback


def plot_cb(pl_module, force_plot: bool):
    idx_epoch = pl_module.current_epoch
    pf = pl_module.plot_freq
    z = pl_module.dataloader.dataset.z
    if pf > 0:
        if (idx_epoch % pf == 0) or force_plot:
            pl_module.visualize(z, 0.0, idx_epoch)


class MyCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        plot_cb(pl_module, True)
        print("Training started.")

    def on_train_end(self, trainer, pl_module):
        print(f"Training finished!")
        plot_cb(pl_module, True)

    def on_epoch_end(self, trainer, pl_module):
        plot_cb(pl_module, False)

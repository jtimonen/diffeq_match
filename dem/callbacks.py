from pytorch_lightning.callbacks import Callback


class MyCallback(Callback):

    def on_train_start(self, trainer, plm):
        print("Training started.")

    def on_train_end(self, trainer, plm):
        print(f"Training finished!")

    def on_epoch_end(self, trainer, pl_module):
        idx_epoch = pl_module.current_epoch
        df = pl_module.draw_freq
        z = pl_module.dataloader.dataset.z
        if df > 0:
            if idx_epoch % df == 0:
                pl_module.visualize(z, 0.0, idx_epoch)

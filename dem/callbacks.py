from pytorch_lightning.callbacks import Callback


class PrintCallback(Callback):

    def on_train_start(self, trainer, plm):
        print("Training started.")

    def on_train_end(self, trainer, plm):
        print(f"Training finished!")

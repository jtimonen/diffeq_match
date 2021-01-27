from pytorch_lightning.callbacks import Callback


class MyCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training started.")

    def on_train_end(self, trainer, pl_module):
        print(f"Training finished!")

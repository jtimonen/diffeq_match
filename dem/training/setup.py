from pytorch_lightning import Trainer, LightningModule


def run_training(module: LightningModule, n_epochs: int, outdir):
    trainer = Trainer(max_epochs=n_epochs, min_epochs=n_epochs, default_root_dir=outdir)
    trainer.fit(module)
    return trainer


class TrainingSetup:
    def __init__(
        self,
        train_dataset,
        valid_dataset=None,
        lr: float = 0.005,
        n_epochs: int = 400,
        outdir="out",
        batch_size: int = 64,
        num_workers: int = 0,
        plot_freq: int = 0,
    ):
        if valid_dataset is None:
            valid_dataset = train_dataset
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.outdir = outdir
        self.plot_freq = plot_freq

    def __repr__(self):
        desc = "<TrainingSetup (lr=%1.4f, n_epochs=%d)>" % (self.lr, self.n_epochs)
        return desc

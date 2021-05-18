from .discriminator import train_occ
from .model import train_model, GAN


# Exports
functions = ["train_occ", "train_model"]
classes = ["GAN"]
__all__ = functions + classes

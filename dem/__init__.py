from dem.modules import *
from dem.utils import *
from dem.plotting import *
from dem.data import *
from dem.training import *

# Version defined here
__version__ = "0.0.5"

# Exports
functions = [
    "create_model",
    "create_discriminator",
    "create_dynamics",
    "load_data_txt",
    "sim",
    "plot_sim",
    "set_device",
    "get_device",
    "session_info",
    "plot_disc_2d",
    "train_occ",
    "train_model",
    "accuracy",
    "split_by_labels",
    "visualize",
]

classes = [
    "GenerativeModel",
    "DynamicModel",
    "PriorInfo",
    "KdeDiscriminator",
    "NeuralDiscriminator",
    "VectorField",
    "StochasticVectorField",
    "GAN",
]
__all__ = functions + classes

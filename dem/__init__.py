from dem.modules import *
from dem.utils import *
from dem.plotting import *
from dem.data import *

# Version defined here
__version__ = "0.0.3"
__pkg_name__ = "dem"

# Exports
functions = [
    "create_model",
    "create_prior_info",
    "load_data_txt",
    "sim",
    "plot_sim",
    "set_device",
    "get_device",
    "session_info",
    "plot_disc_2d",
]

classes = [
    "GenModel",
    "PriorInfo",
    "KdeDiscriminator",
    "NeuralDiscriminator",
    "VectorField",
    "StochasticVectorField",
]
__all__ = functions + classes

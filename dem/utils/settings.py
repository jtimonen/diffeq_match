import torch
import sys

# Global variables
_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def set_device(device_name):
    """Set a global device for torch.
    :param device_name: Name of the device, such as 'cpu' or 'cuda'.
    :type device_name: str
    """
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda is not available!")
    dev = torch.device(device_name)
    global _DEVICE
    _DEVICE = dev


def get_device():
    """Get current torch device.
    :return: device
    :rtype: torch.device
    """
    global _DEVICE
    return _DEVICE


def session_info(skip_cuda: bool = False):
    """Print version info of relevant dependencies and set output path.
    :param skip_cuda: Should CUDA information be skipped?
    :type skip_cuda: bool
    """
    print(session_info_(skip_cuda))


def session_info_(skip_cuda: bool = False):
    """Get version info of relevant dependencies, and set data and output paths.
    :param skip_cuda: Should CUDA information be skipped?
    :type skip_cuda: bool
    """
    ver = sys.version_info
    ver = str(ver[0]) + "." + str(ver[1]) + "." + str(ver[2])
    msg = "Using "
    msg = msg + " python-" + ver
    msg = msg + ", torch-" + str(torch.__version__)
    msg += "\n - device: " + str(get_device())
    if not skip_cuda:
        msg = msg + "\n" + cuda_info()
    return msg


def cuda_info():
    """Show CUDA device info."""
    if torch.cuda.is_available():
        msg = "CUDA info:"
        msg += "\n - cudnn version: " + str(torch.backends.cudnn.version())
        msg += "\n - number of CUDA devices:" + str(torch.cuda.device_count())
        msg += "\n - active CUDA device:" + str(torch.cuda.current_device())
    else:
        msg = " - CUDA not available"
    return msg

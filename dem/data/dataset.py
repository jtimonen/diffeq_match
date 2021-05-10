import numpy as np
from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    """Torch dataset for training using data consisting of a single numpy array."""

    def __init__(self, x: np.ndarray):
        super(NumpyDataset, self).__init__()
        self.x = x
        self.N = x.shape[0]
        self.D = x.shape[1]

    def __len__(self) -> int:
        """Get number of data samples."""
        return self.N

    def __getitem__(self, idx: int) -> np.ndarray:
        """Get a data sample for a given key.
        :param int idx: data point index
        """
        x = self.x[idx, :].astype(np.float32)
        return x

    def __repr__(self):
        desc = "A NumpyDataset with %d observations and %d dimensions." % (
            self.N,
            self.D,
        )
        return desc

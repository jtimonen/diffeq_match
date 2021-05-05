import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """A torch dataset."""

    def __init__(self, z: np.ndarray):
        super(MyDataset, self).__init__()
        self.z = z
        self.N = z.shape[0]
        self.D = z.shape[1]

    def __len__(self):
        """Get number of data samples."""
        return self.N

    def __getitem__(self, idx: int):
        """Get a data sample for a given key.
        :param int idx: data point index
        """
        return self.z[idx, :]

    def __repr__(self):
        """Get description of a MyDataset instance."""
        desc = "A Dataset with %d observations and %d dimensions." % (
            self.N,
            self.D,
        )
        return desc


class ClassificationDataset(Dataset):
    """Torch dataset for supervised training of a model that classifies
    points represented by rows of a numpy array.

    :param x: the data to be classified, shape (N, D)
    :param labels: true class labels, length N
    """

    def __init__(self, x: np.ndarray, labels: np.ndarray):
        super(ClassificationDataset, self).__init__()
        self.x = x
        self.N = x.shape[0]
        self.D = x.shape[1]
        self.labels = labels

    def __len__(self):
        """Get number of data samples."""
        return self.N

    def __getitem__(self, idx: int):
        """Get a data sample for a given key.
        :param int idx: data point index
        """
        return self.x[idx, :], self.labels[idx]

    def __repr__(self):
        desc = "A ClassificationDataset with %d observations and %d dimensions." % (
            self.N,
            self.D,
        )
        return desc

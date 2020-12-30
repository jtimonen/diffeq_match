import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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


def create_dataloader(
    dataset: torch.utils.data.Dataset, batch_size=None, num_workers=None
):
    """Create a torch data loader."""
    # Split to training and validation sets
    N = len(dataset)
    if batch_size is None:
        batch_size = N

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return loader

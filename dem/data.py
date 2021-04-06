import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_data_txt(parent_dir, verbose=True):
    """Load data (with dynverse annotations) from .txt files."""
    p1 = os.path.join(parent_dir, "latent.txt")
    p2 = os.path.join(parent_dir, "init_indices.txt")
    p3 = os.path.join(parent_dir, "start_idx.txt")
    p4 = os.path.join(parent_dir, "end_idx.txt")
    z_data = np.loadtxt(p1)
    init_indices = np.loadtxt(p2).astype(int)
    start_idx = int(np.loadtxt(p3))
    end_idx = int(np.loadtxt(p4))
    z0 = np.array(z_data[init_indices, :])
    if verbose:
        print("loaded data from:", parent_dir)
        print(" * z_data shape:", z_data.shape)
        print(" * z0 shape", z0.shape)
        print(" * start_idx:", start_idx)
        print(" * end_idx:", end_idx)
    return z_data, z0, start_idx, end_idx


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
    dataset: torch.utils.data.Dataset, batch_size=None, num_workers=None, shuffle=True
):
    """Create a torch data loader."""
    # Split to training and validation sets
    N = len(dataset)
    if batch_size is None:
        batch_size = N

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return loader

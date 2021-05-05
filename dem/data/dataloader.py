import os
import numpy as np
import torch
from torch.utils.data import DataLoader


def load_data_txt(parent_dir, verbose=True):
    """Load data (with dynverse annotations) from .txt files."""
    p1 = os.path.join(parent_dir, "latent.txt")
    p2 = os.path.join(parent_dir, "init_indices.txt")
    p3 = os.path.join(parent_dir, "start_idx.txt")
    p4 = os.path.join(parent_dir, "end_idx.txt")
    z_data = np.loadtxt(p1)
    init_indices = np.loadtxt(p2).astype(int)
    start_idx = np.loadtxt(p3).astype(int)
    end_idx = np.loadtxt(p4).astype(int)
    z0 = np.array(z_data[init_indices, :])
    if verbose:
        print("loaded data from:", parent_dir)
        print(" * z_data shape:", z_data.shape)
        print(" * z0 shape", z0.shape)
        print(" * start_idx:", start_idx)
        print(" * end_idx:", end_idx)
    return z_data, z0, start_idx, end_idx


def create_dataloader(
    dataset: torch.utils.data.Dataset, batch_size=None, num_workers=None, shuffle=True
):
    """Create a torch data loader."""
    N = len(dataset)
    if batch_size is None:
        batch_size = N
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return loader

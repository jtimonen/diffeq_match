import numpy as np
import torch


def pastep(s1: str, s2: str):
    """Paste two string so that the latter is in parentheses."""
    return s1 + " (" + s2 + ")"


def pasteq(s1: str, s2: str):
    """Paste two string so that the latter is in quotes."""
    return s1 + " '" + s2 + "'"


def dot_torch(a, b):
    """Pairwise dot products of the rows of a and b. Matrices a and b must have equal
    shape.
    """
    N = a.shape[0]
    D = a.shape[1]
    return torch.bmm(a.view(N, 1, D), b.view(N, D, 1)).view(N, -1)


def logit_torch(p, eps=1e-8):
    """Logit transformation using torch."""
    return torch.log((p + eps) / (1 - p + eps))


def find_closest_point(x, x_candidates):
    """Find point closest to x."""
    dist = ((x - x_candidates) ** 2).sum(axis=1)
    idx = np.argmin(dist)
    return x_candidates[idx, :]


def normalize_torch(x: torch.tensor):
    """Normalize expression (torch tensor input and output)."""
    s = x.sum(axis=1)
    x = 100 * x / s[:, None]
    return torch.log1p(x)


def normalize_numpy(x):
    """Normalize expression (numpy array input and output)."""
    x = torch.from_numpy(x).float()
    x = normalize_torch(x)
    return x.detach().numpy()


def array_info(X):
    info = str(round(X.nbytes / 1e6, 3)) + " MB, dtype = " + str(X.dtype)
    return info


def create_grid_around(z, M: int, scaling: float = 0.1):
    """Create a uniform rectangular grid around points *z*.

    :param z: a numpy array of shape *[n_points, d]*
    :type z: np.ndarray
    :param M: number of points per dimension
    :type M: int
    :param scaling: How much larger should the grid be than the range of *z* for each dimension.
        If this is zero, grid is exactly the size of the data range.
    :type scaling: float
    :return: a numpy array of shape *[M^d, d]*
    :rtype: np.ndarray
    """
    umin = np.amin(z, axis=0)
    umax = np.amax(z, axis=0)
    D = len(umax)
    LS = list()
    for d in range(0, D):
        h = scaling * (umax[d] - umin[d])
        LS = LS + [np.linspace(umin[d] - h, umax[d] + h, M)]
    xs_ = np.meshgrid(*LS)
    U_grid = np.array([x.T.flatten() for x in xs_]).T
    return U_grid


def create_grid_close_to_data(z, M: int, scaling: float = 0.1, max_dist: float = 0.2):
    """Create a grid close to data points *z*.

    :param z: a numpy array of shape *[n_points, d]*
    :type z: np.ndarray
    :param M: number of points per dimension
    :type M: int
    :param scaling: How much larger should the grid be than the range of *z* for each dimension.
        If this is zero, grid is exactly the size of the data range.
    :type scaling: float
    :param max_dist: maximum allowed distance from nearest data point
    :type max_dist: float

    :return: a numpy array of shape *[s, d]*
    :rtype: np.ndarray
    """
    U_grid = create_grid_around(z, M, scaling)
    dist = closest_point_distance(U_grid, z)
    U_grid = U_grid[dist < max_dist, :]
    return U_grid


def assert_numeric(x):
    """Test if tensor contains Inf, -Inf, or NaN.
    :raises: RuntimeError
    """
    if torch.isnan(x).any():
        raise RuntimeError("Tensor contains NaN!")
    if (x == -float("inf")).any():
        raise RuntimeError("Tensor contains -Inf!")
    if (x == float("inf")).any():
        raise RuntimeError("Tensor contains +Inf!")


def assert_positive(x):
    """Test if tensor contains only positive values.
    :raises: RuntimeError
    """
    if (x <= 0).any():
        raise RuntimeError("Tensor contains values <= 0!")


def determine_nrows_ncols(nplots: int):
    """Determine number of rows and columns a grid of subplots.

    :param nplots: total number of subplots
    :type nplots: int
    """
    if nplots < 4:
        ncols = nplots
    elif nplots < 5:
        ncols = 2
    elif nplots < 10:
        ncols = 3
    else:
        ncols = 4
    nrows = int(np.ceil(nplots / ncols))
    return nrows, ncols

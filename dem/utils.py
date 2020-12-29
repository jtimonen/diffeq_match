import numpy as np


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

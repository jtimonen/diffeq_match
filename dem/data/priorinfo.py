import numpy as np


def create_prior_info(z_init, z_terminal):
    """Create prior info about generative process."""
    if (z_init is None) and (z_terminal is None):
        raise RuntimeError("z_init and z_terminal cannot both be None!")
    if z_init is None:
        z = z_terminal
        t = np.ones(z.shape[0])
    else:
        z = z_init
        t = np.zeros(z.shape[0])
    return PriorInfo(z, t)


class PriorInfo:
    """Encodes prior information about observation times.

    :param z: array with shape (N, D)
    :param t: array with length N
    """

    def __init__(self, z: np.ndarray, t: np.ndarray):
        self.N = z.shape[0]
        self.D = z.shape[1]
        assert len(t) == self.N, "t must have length equal to number of rows in z"
        self.z = z
        self.t = t

    def draw_points(self, num_points: int):
        """Draw observations from known time points.

        :param num_points: number of points to draw
        """
        idx = np.random.choice(self.N, size=num_points, replace=True)
        z_draw = self.z[idx, :]
        t_draw = self.t[idx]
        return z_draw, t_draw

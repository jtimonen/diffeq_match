from sklearn.metrics import accuracy_score
import numpy as np


def accuracy(y_true, y_pred, prob=False):
    if prob:
        pred_labels = (y_pred > 0.5).astype(float)
    else:
        pred_labels = y_pred
    return accuracy_score(y_true=y_true, y_pred=pred_labels)


def split_by_labels(x, labels):
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    x0 = x[idx0, :]
    x1 = x[idx1, :]
    return x0, x1


def reshape_traj(z_traj):
    n_timepoints = z_traj.shape[0]
    n_samples = z_traj.shape[1]
    n_dimensions = z_traj.shape[2]
    return z_traj.view(n_timepoints * n_samples, n_dimensions)

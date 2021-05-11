from sklearn.metrics import accuracy_score
import numpy as np
import torch


def tensor_to_numpy(x: torch.Tensor):
    return x.cpu().detach().numpy()


def target_labels(N1, N0, dtype=np.uint8):
    y_target = np.array(N1 * [1] + N0 * [0], dtype=dtype)
    return y_target


def create_classification(x_real: torch.Tensor, x_noisy: torch.Tensor):
    N1 = x_real.shape[0]
    N0 = x_noisy.shape[0]
    x = torch.vstack((x_real, x_noisy))
    y_target = target_labels(N1, N0, dtype=np.float32)
    y_target = torch.from_numpy(y_target).to(x.device)
    return x, y_target


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

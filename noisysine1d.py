import torch
import numpy as np
from .utils import xdata, noise


def noisysine1d(x, sigma: float = 0.1):
    return torch.sin(2 * np.pi * x) + noise(x.shape, sigma) * (1.5 - torch.abs(torch.sin(2 * np.pi * x)))


def load_noisysine1d(train_dataset_size: int = 2**10, test_dataset_size: int = 2**10, **kwargs):
    x_train = xdata(dataset_size=train_dataset_size, generate_random=True, **kwargs)
    y_train = noisysine1d(x_train)
    x_test = xdata(dataset_size=test_dataset_size, generate_random=True, **kwargs)
    y_test = noisysine1d(x_test)

    input_shape, output_shape = x_train.shape[-1], y_train.shape[-1]
    return x_train, y_train, x_test, y_test, input_shape, output_shape

from torchvision import datasets
import os
import torch
import numpy as np
from torch.utils.data import random_split
from torchvision import transforms

from .utils import to_categorical, dataset_points_to_paths


def load_cifar10(dataset_size_train: int = 50000, dataset_size_test: int = 10000, flatten: bool = False,
                 debug: bool = False, generate_paths: bool = False, **kwargs):
    img_rows, img_cols, img_channels = 32, 32, 3

    file_path = '{}/{}'.format(os.path.dirname(__file__), '/cifar10')
    train_dataset = datasets.CIFAR10(file_path, train=True, download=True,
                                     transform=transforms.Compose([transforms.ToTensor(), ]))
    test_dataset = datasets.CIFAR10(file_path, train=False, download=True,
                                    transform=transforms.Compose([transforms.ToTensor(), ]))

    x_train = torch.from_numpy(train_dataset.data) / 255
    y_train = to_categorical(train_dataset.targets, 10)
    x_test = torch.from_numpy(test_dataset.data) / 255
    y_test = to_categorical(test_dataset.targets, 10)

    # create grid of path_length * classes for plotting
    mask = []
    path_length_per_class = kwargs['path_length'] // 10
    for idx in range(10):
        mask.append(torch.where(y_test[:, idx] == 1)[0][:path_length_per_class])
    mask = torch.cat(mask)
    x_plot = x_test[mask]
    y_plot = y_test[mask]

    if debug:
        debug_points = 2 ** 12
        print('In DEBUG mode training and test points are reduced to: {}'.format(debug_points))
        x_train, y_train = x_train[0:debug_points], y_train[0:debug_points]
        x_test, y_test = x_test[0:debug_points], y_test[0:debug_points]

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols * img_channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols * img_channels)
        x_plot = x_plot.reshape(x_plot.shape[0], img_rows * img_cols * img_channels)

        if generate_paths:
            x_train, y_train = dataset_points_to_paths(x_train, y_train, **kwargs)
            x_test, y_test = dataset_points_to_paths(x_test, y_test, **kwargs)
            x_plot, y_plot = dataset_points_to_paths(x_plot, y_plot, **kwargs)

        input_shape = x_train.shape[-1]
        output_shape = y_train.shape[-1]
    else:
        x_train, x_test, x_plot = x_train.moveaxis(-1, 1), x_test.moveaxis(-1, 1), x_plot.moveaxis(-1, 1)

        input_shape = x_train.shape[-3:]
        output_shape = y_train.shape[-1]
    return x_train, y_train, x_test, y_test, x_plot, y_plot, input_shape, output_shape

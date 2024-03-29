from torchvision import datasets
import os
import torch
from torch.utils.data import random_split
from torchvision import transforms

from .utils import to_categorical, dataset_points_to_paths


def load_fashion_mnist(dataset_size_train: int = 60000, dataset_size_test: int = 10000, flatten: bool = True,
                       debug: bool = False, generate_paths: bool = False, **kwargs):
    img_rows, img_cols = 28, 28

    file_path = '{}/{}'.format(os.path.dirname(__file__), '/fashion_mnist')
    train_dataset = datasets.FashionMNIST(file_path, train=True, download=True,
                                          transform=transforms.Compose([transforms.ToTensor(), ]))
    test_dataset = datasets.FashionMNIST(file_path, train=False, download=True,
                                          transform=transforms.Compose([transforms.ToTensor(), ]))

    x_train = train_dataset.data / 255
    y_train = to_categorical(train_dataset.targets, 10)
    x_test = test_dataset.data / 255
    y_test = to_categorical(test_dataset.targets, 10)

    x_plot = torch.ones(x_train.shape).fill_(torch.nan)
    y_plot = torch.ones(y_train.shape).fill_(torch.nan)

    if debug:
        debug_points = 2 ** 12
        print('In DEBUG mode training and test points are reduced to: {}'.format(debug_points))
        x_train, y_train = x_train[0:debug_points], y_train[0:debug_points]
        x_test, y_test = x_test[0:debug_points], y_test[0:debug_points]

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)
        x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols)
        x_plot = x_plot.reshape(x_plot.shape[0], img_rows * img_cols)

        if generate_paths:
            x_train, y_train = dataset_points_to_paths(x_train, y_train, **kwargs)
            x_test, y_test = dataset_points_to_paths(x_test, y_test, **kwargs)
            x_plot, y_plot = dataset_points_to_paths(x_plot, y_plot, **kwargs)

        input_shape = x_train.shape[-1]
        output_shape = y_train.shape[-1]
        return x_train, y_train, x_test, y_test, x_plot, y_plot, input_shape, output_shape
    else:
        raise NotImplementedError

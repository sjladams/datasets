from torchvision import datasets
import os
import torch
import numpy as np
from torch.utils.data import random_split
from torchvision import transforms

from .utils import to_categorical, dataset_points_to_paths


def load_cifar10(dataset_size_train: int = 52500, dataset_size_test: int = 7500, flatten: bool = False,
                 debug: bool = False, generate_paths: bool = False, **kwargs):
    img_rows, img_cols, img_channels = 32, 32, 3

    file_path = '{}/{}'.format(os.path.dirname(__file__), '/cifar10')
    cifar10 = datasets.CIFAR10(file_path, train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),]))

    train_points = int(cifar10.data.shape[0] * (dataset_size_train / (dataset_size_train + dataset_size_test)))
    test_points = int(cifar10.data.shape[0] * (dataset_size_test / (dataset_size_train + dataset_size_test)))

    cifar10_train, cifar10_test = random_split(cifar10, [train_points, test_points])

    x_train = torch.tensor(cifar10.data[cifar10_train.indices] / 255).type(torch.float32)
    y_train = to_categorical(np.array(cifar10.targets)[cifar10_train.indices], 10)
    x_test = torch.tensor(cifar10.data[cifar10_test.indices] / 255).type(torch.float32)
    y_test = to_categorical(np.array(cifar10.targets)[cifar10_test.indices], 10)

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


# def load_cifar(channels, img_rows=32, img_cols=32):
    # path = "{}/examples/cifar/data".format(os.getcwd())
    # trainset = datasets.CIFAR10(root=path, train=True, download=True,
    #                             transform=transforms.Compose([transforms.ToTensor(),]))
    # x_train = torch.from_numpy(trainset.data) / 255
    # y_train = to_categorical(torch.tensor(trainset.targets), 10)
    #
    # # testset = datasets.CIFAR10(root=path, train=False, download=True,
    # #                            transform=transforms.Compose([transforms.ToTensor(),]))
    # # x_test = torch.from_numpy(testset.data) / 255
    # # y_test = to_categorical(torch.tensor(testset.targets), 10)
    # #
    # # np.save(f"{path}/x_test_deepmind", testset.data / 255)
    # # np.save(f"{path}/y_test_deepmind", testset.targets)
    #
    # x_test = torch.from_numpy(pickle_load(f"{path}/x_test_deepmind.npy")).type(torch.float32)
    # y_test = to_categorical(pickle_load(f"{path}/y_test_deepmind.npy"), 10).type(torch.float32)
    #
    # # if channels == "first":
    # #     x_train = x_train.reshape(-1, 32 * 32 *3)
    # #     x_test = x_test.reshape(-1, 32 * 32 *3)
    # x_train = x_train.moveaxis(3, 1)
    # x_test = x_test.moveaxis(3, 1)
    #
    # input_shape = x_train.shape[1:]
    # num_classes = 10
    # return x_train, y_train, x_test, y_test, input_shape, num_classes

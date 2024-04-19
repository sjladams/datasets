from torch.utils.data import DataLoader
import torch
import numpy as np
import random

from .regr1d import load_regr1d
from .linear import load_linear
from .noisysine1d import load_noisysine1d
from .noisysine2d import load_noisysine2d
from .kin8nm import load_kin8nm
from .mnist import load_mnist
from .fashion_mnist import load_fashion_mnist
from .snelson1d import load_snelson1d
from .boston import load_boston
from .concrete import load_concrete
from .energy import load_energy
from .naval import load_naval
from .power import load_power
from .protein import load_protein
from .wine import load_wine
from .luca import load_luca
from .luca2d import load_luca2d
from .cifar10 import load_cifar10
from .example1d import load_example1d
from .uva_tutorial import load_uva_tutorial


def data_loaders(dataset_name, batch_size_train: int, batch_size_test: int, shuffle=False, sort=False, **kwargs):
    if sort:
        shuffle = False

    x_train, y_train, x_test, y_test, x_plot, y_plot, input_shape, output_shape = load_dataset(
        dataset_name=dataset_name, shuffle=shuffle, sort=sort, **kwargs)

    train_ds = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size_train, shuffle=shuffle,
                              worker_init_fn=np.random.seed(0), num_workers=0)

    test_ds = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size_test, shuffle=shuffle,
                             worker_init_fn=np.random.seed(0), num_workers=0)

    plot_ds = torch.utils.data.TensorDataset(x_plot, y_plot)

    return train_loader, test_loader, plot_ds, input_shape, output_shape


def get_data_loader(dataset_name: str):
    if dataset_name == 'noisysine1d':
        return load_noisysine1d
    elif dataset_name == 'noisysine2d':
        return load_noisysine2d
    elif dataset_name == 'linear1d':
        return load_linear
    elif dataset_name == 'kin8nm':
        return load_kin8nm
    elif dataset_name == "regr1d":
        return load_regr1d
    elif dataset_name == 'mnist':
        return load_mnist
    elif dataset_name == 'fashion_mnist':
        return load_fashion_mnist
    elif dataset_name == 'snelson1d':
        return load_snelson1d
    elif dataset_name == 'boston':
        return load_boston
    elif dataset_name == 'concrete':
        return load_concrete
    elif dataset_name == 'energy':
        return load_energy
    elif dataset_name == 'naval':
        return load_naval
    elif dataset_name == 'power':
        return load_power
    elif dataset_name == 'protein':
        return load_protein
    elif dataset_name == 'wine':
        return load_wine
    elif dataset_name == 'luca':
        return load_luca
    elif dataset_name == 'luca2d':
        return load_luca2d
    elif dataset_name == 'cifar10':
        return load_cifar10
    elif dataset_name == 'example1d':
        return load_example1d
    elif dataset_name == 'uva_tutorial':
        return load_uva_tutorial
    else:
        raise AssertionError("\nDataset not available.")


def load_dataset(dataset_name, shuffle=False, sort=False, generate_paths=False, **kwargs):
    data_loader = get_data_loader(dataset_name)
    x_train, y_train, x_test, y_test, x_plot, y_plot, input_shape, output_shape = data_loader(
            generate_paths=generate_paths, **kwargs)

    # print('x_train shape =', x_train.shape, '\nx_test shape =', x_test.shape)
    # print('y_train shape =', y_train.shape, '\ny_test shape =', y_test.shape)
    # print('y_plot shape =', y_plot.shape, '\ny_plot shape =', y_plot.shape)

    if shuffle is True:
        random.seed(0)
        idxs_train = torch.randperm(len(x_train))
        x_train, y_train = (x_train[idxs_train], y_train[idxs_train])
        idxs_test = torch.randperm(len(x_test))
        x_test, y_test = (x_test[idxs_test], y_test[idxs_test])

    if sort is True and not generate_paths:
        mask_train = x_train.sort(dim=0).indices[:, 0]  # order test data w.r.t. 1st dimension
        x_train, y_train = x_train[mask_train], y_train[mask_train]
        mask_test = x_test.sort(dim=0).indices[:, 0]  # order test data w.r.t. 1st dimension
        x_test, y_test = x_test[mask_test], y_test[mask_test]

    if not generate_paths:
        mask_plot = x_plot.sort(dim=0).indices[:, 0]  # order test data w.r.t. 1st dimension
        x_plot, y_plot = x_plot[mask_plot], y_plot[mask_plot]

    return x_train, y_train, x_test, y_test, x_plot, y_plot, input_shape, output_shape

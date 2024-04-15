import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from .utils import post_process_data, pickle_dump


def load_example1d(dataset_size_train: int, dataset_size_test: int, data_specs: dict,
                   generate_ood: bool = False, **kwargs):
    """

    """
    path = '{}/{}'.format(os.path.dirname(__file__), '/example1d/')
    inputs_file_path = os.path.join(path, 'train_inputs')
    outputs_file_path = os.path.join(path, 'train_outputs')
    if os.path.exists(inputs_file_path) and os.path.exists(outputs_file_path):
        x_raw = np.loadtxt(inputs_file_path)[:, None]
        y_raw = np.loadtxt(outputs_file_path)[:, None]
    else:
        raise NotImplementedError('data file not available')

    gap = data_specs['gap']
    mask = np.logical_and(x_raw > gap[0], x_raw < gap[1])
    x = x_raw[~mask][..., None]
    y = y_raw[~mask][..., None]

    x_train = x[:dataset_size_train, :]
    y_train = y[:dataset_size_train]

    x_test = x[-dataset_size_test:, :]
    y_test = y[-dataset_size_test:]

    window = x.max() - x.min()
    ood_expansion = window * 0.25

    # use uniform grid for plot values
    nr_plot_points = 100
    x_plot = np.linspace(x.min() - ood_expansion, x.max() + ood_expansion, nr_plot_points)[..., None]
    y_plot = np.full((nr_plot_points, 1), np.nan)

    x_scalar = StandardScaler()
    x_scalar.fit(x_raw)
    x_scalar.mean_ = np.array([0.])

    y_scalar = StandardScaler()
    y_scalar.fit(y_raw)

    if generate_ood:
        x_train = np.linspace(x.min() - ood_expansion, x.max() + ood_expansion, dataset_size_train)[..., None]
        y_train = np.full((dataset_size_train, 1), np.nan)

        x_test = np.linspace(x.min() - ood_expansion, x.max() + ood_expansion, dataset_size_test)[..., None]
        y_test = np.full((dataset_size_test, 1), np.nan)

    scale = True
    x_train, y_train = post_process_data(x_scalar, y_scalar, x_train, y_train, scale=scale, **kwargs)
    x_test, y_test = post_process_data(x_scalar, y_scalar, x_test, y_test, scale=scale, **kwargs)
    x_plot, y_plot = post_process_data(x_scalar, y_scalar, x_plot, y_plot, scale=scale, **kwargs)

    input_shape = x_train.shape[-1]
    output_shape = y_train.shape[-1]

    ## save data
    modified_data_file_path = os.path.join(path, 'modified_data')
    if not generate_ood and not os.path.exists(f"{modified_data_file_path}.pickle"):
        pickle_dump({'train': {'x': x_train, 'y': y_train},
                     'test': {'x': x_test, 'y': y_test},
                     'plot': {'x': x_plot, 'y': y_plot}}, modified_data_file_path)

    return x_train, y_train, x_test, y_test, x_plot, y_plot, input_shape, output_shape

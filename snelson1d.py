import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from .utils import post_process_data, pickle_dump


def load_snelson1d(len_train_dataset: int, len_test_dataset: int, data_specs: dict,
                   generate_ood: bool = False, **kwargs):
    """
    Load Edward Snelson's 1d regression data set [@snelson2006fitc].
    It contains 200 examples of a few oscillations of an example function. It has
    seen extensive use as a toy dataset for illustrating qualitative behaviour of
    Gaussian process approximations.

    :param len_train_dataset:
    :param len_test_dataset:
    :param data_specs:
    :param generate_ood: if true, return samples from dataset, else generate (grid of) samples
    :param kwargs:
    """
    path = '{}/{}'.format(os.path.dirname(__file__), '/snelson/')
    inputs_file_path = os.path.join(path, 'train_inputs')
    outputs_file_path = os.path.join(path, 'train_outputs')
    if os.path.exists(inputs_file_path) and os.path.exists(outputs_file_path):
        x_raw = np.loadtxt(inputs_file_path)[:, None]
        y_raw = np.loadtxt(outputs_file_path)[:, None]
    else:
        raise NotImplementedError('data file not available')

    gap = data_specs['gap']
    # as in https://arxiv.org/abs/1906.01930 remove all points between 3 and 5:
    mask = np.logical_and(x_raw > gap[0], x_raw < gap[1]) # x > 1.5, x < 3. # x > 3.6, x < 5. # x > 1.5, x < 3.5
    x = x_raw[~mask][..., None]
    y = y_raw[~mask][..., None]

    x_train = x[:len_train_dataset, :]
    y_train = y[:len_train_dataset]

    x_test = x[-len_test_dataset:, :]
    y_test = y[-len_test_dataset:]

    window = x.max() - x.min()
    ood_expansion = window * 0.25

    # use uniform grid for plot values
    nr_plot_points = 1000
    x_plot = np.linspace(round(x.min() - ood_expansion), round(x.max() + ood_expansion), nr_plot_points)[..., None]
    y_plot = np.full((nr_plot_points, 1), np.nan)

    pre_processer_x = StandardScaler()
    pre_processer_x.fit(x_raw)

    pre_processer_y = StandardScaler()
    pre_processer_y.fit(y_raw)

    if generate_ood:
        x_train = np.linspace(x.min() - ood_expansion, x.max() + ood_expansion, len_train_dataset)[..., None]
        y_train = np.full((len_train_dataset, 1), np.nan)

        x_test = np.linspace(x.min() - ood_expansion, x.max() + ood_expansion, len_test_dataset)[..., None]
        y_test = np.full((len_test_dataset, 1), np.nan)

    scale = True
    x_train, y_train = post_process_data(pre_processer_x, pre_processer_y, x_train, y_train, scale=scale, **kwargs)
    x_test, y_test = post_process_data(pre_processer_x, pre_processer_y, x_test, y_test, scale=scale, **kwargs)
    x_plot, y_plot = post_process_data(pre_processer_x, pre_processer_y, x_plot, y_plot, scale=scale, **kwargs)

    input_size = x_train.shape[-1]
    output_size = y_train.shape[-1]

    ## save data
    modified_data_file_path = os.path.join(path, 'modified_data')
    if not generate_ood and not os.path.exists(f"{modified_data_file_path}.pickle"):
        pickle_dump({'train': {'x': x_train, 'y': y_train},
                     'test': {'x': x_test, 'y': y_test},
                     'plot': {'x': x_plot, 'y': y_plot}}, modified_data_file_path)

    return x_train, y_train, x_test, y_test, x_plot, y_plot, input_size, output_size

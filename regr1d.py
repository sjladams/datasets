# import gp.utils
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from .utils import post_process_data


def make_random_gap(X, gap_ratio=0.2):
    a,b = X.min(),X.max()
    gap_a = a + np.random.rand() * (b-a)*(1-gap_ratio)
    gap_b = gap_a + (b-a)*gap_ratio
    idx = np.logical_and(gap_a<X, X<gap_b)
    if gap_a-a > b-gap_b:
        X[idx] = a + np.random.rand(idx.sum()) * (gap_a-a)
    else:
        X[idx] = gap_b + np.random.rand(idx.sum()) * (b-gap_b)


def gp_sample(X, ampl=1, leng=1, sn2=0.1):
    n, x = X.shape[0], X / leng
    sum_xx = np.sum(x*x, 1).reshape(-1, 1).repeat(n, 1)
    D = sum_xx + sum_xx.transpose() - 2 * np.matmul(x, x.transpose())
    C = ampl**2 * np.exp(-0.5 * D) + np.eye(n) * sn2
    return np.random.multivariate_normal(np.zeros(n), C).reshape(-1, 1)


def get_regr1d_dataset(l: float, u: float):
    np.random.seed(1)
    N = 64

    # Generate data
    X = np.random.rand(N, 1) * (u - l) + l
    make_random_gap(X, gap_ratio=0.4)
    y = gp_sample(X, ampl=1.6, leng=1.8)
    return X, y


# def gpr_generator(x: torch.Tensor, dataset_size: int, generate_paths: bool, **kwargs):
#     if dataset_size < 2**5 and not generate_paths:
#         gpr = gp.utils.construct_gp(x, torch.zeros((dataset_size, 1)), **kwargs)
#         return gpr.sample(x, num_samples=1).reshape(dataset_size, 1)
#     else:
#         print('generating y_train-data is too expensive, set to zeros')
#         return torch.zeros(x.shape[:-1] + (1,))


def load_regr1d(dataset_size_train: int, dataset_size_test: int, data_specs: dict,
                   generate_ood: bool = False, nr_plot_points: int = 1000, **kwargs):
    """
    1d regression dataset from https://arxiv.org/abs/2011.12829

    :param dataset_size_train:
    :param dataset_size_test:
    :param data_specs:
    :param generate_ood: if true, return samples from dataset, else generate (grid of) samples
    :param l: lower bound dataset
    :param u: upper bound dataset
    :param kwargs:
    """
    path = '{}/{}'.format(os.path.dirname(__file__), '/regr1d/')
    inputs_file_path = os.path.join(path, 'train_inputs')
    outputs_file_path = os.path.join(path, 'train_outputs')
    if os.path.exists(inputs_file_path) and os.path.exists(outputs_file_path):
        x = np.loadtxt(inputs_file_path)[:, None]
        y = np.loadtxt(outputs_file_path)[:, None]
    else:
        x, y = get_regr1d_dataset(data_specs['l'], data_specs['u'])
        np.savetxt(inputs_file_path, x)
        np.savetxt(outputs_file_path, y)

    x_train = x[:dataset_size_train, :]
    y_train = y[:dataset_size_train]

    x_test = x[-dataset_size_test:, :]
    y_test = y[-dataset_size_test:]

    # use uniform grid for plot values
    x_plot = np.linspace(data_specs['l'] - 5., data_specs['u'] + 5., nr_plot_points)[..., None]
    y_plot = np.full((nr_plot_points, 1), np.nan)

    pre_processer_x = StandardScaler()
    pre_processer_x.fit(x_train)
    pre_processer_x.mean_ -= 3. # centralise around gap

    pre_processer_y = StandardScaler()
    pre_processer_y.fit(y_train)

    if generate_ood:
        x_train = np.linspace(data_specs['l'] - 5., data_specs['u'] + 5., dataset_size_train)[..., None]
        y_train = np.full((dataset_size_train, 1), np.nan)

        x_test = np.linspace(data_specs['l'] - 5., data_specs['u'] + 5., dataset_size_test)[..., None]
        y_test = np.full((dataset_size_test, 1), np.nan)

    scale = True
    x_train, y_train = post_process_data(pre_processer_x, pre_processer_y, x_train, y_train, scale=scale, **kwargs)
    x_test, y_test = post_process_data(pre_processer_x, pre_processer_y, x_test, y_test, scale=scale, **kwargs)
    x_plot, y_plot = post_process_data(pre_processer_x, pre_processer_y, x_plot, y_plot, scale=scale, **kwargs)

    input_shape = x_train.shape[-1]
    output_shape = y_train.shape[-1]
    return x_train, y_train, x_test, y_test, x_plot, y_plot, input_shape, output_shape

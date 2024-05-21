import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from .utils import post_process_data, pickle_dump


def make_random_gap(X, gap_ratio=0.2):
    a,b = X.min(),X.max()
    gap_a = a + np.random.rand() * (b-a)*(1-gap_ratio)
    gap_b = gap_a + (b-a)*gap_ratio
    idx = np.logical_and(gap_a<X, X<gap_b)
    if gap_a-a > b-gap_b:
        X[idx] = a + np.random.rand(idx.sum()) * (gap_a-a)
    else:
        X[idx] = gap_b + np.random.rand(idx.sum()) * (b-gap_b)


def gp_sample(X, ampl: float = 1, leng: float = 1, sn2: float = 0.1):
    n, x = X.shape[0], X / leng
    sum_xx = np.sum(x*x, 1).reshape(-1, 1).repeat(n, 1)
    D = sum_xx + sum_xx.transpose() - 2 * np.matmul(x, x.transpose())
    C = ampl**2 * np.exp(-0.5 * D) + np.eye(n) * sn2
    return np.random.multivariate_normal(np.zeros(n), C).reshape(-1, 1)


def load_synthetic1d(len_train_dataset: int = 64, len_test_dataset: int = 64, dataset_size_plot: int = 100, 
                     generate_ood: bool = False, **kwargs):
    """
    Dataset introduced in: All You Need is a Good Functional Prior for Bayesian Deep Learning, B. Tran, S. Rossi,
    Milios et al.
    :param len_train_dataset:
    :param len_test_dataset:
    :param generate_ood:
    :param kwargs:
    :return:
    """

    # generate data as in paper
    np.random.seed(1)
    a, b = -10, 10
    dataset_size = 64
    x = np.random.rand(dataset_size, 1) * (b - a) + a
    make_random_gap(x, gap_ratio=0.4)
    y = gp_sample(x, ampl=1.6, leng=1.8)

    x_train, y_train = x, y
    x_test, y_test = x.copy(), y.copy()

    x_train = x_train[:len_train_dataset, :]
    y_train = y_train[:len_train_dataset]

    x_test = x_test[-len_test_dataset:, :]
    y_test = y_test[-len_test_dataset:]

    x_plot = np.linspace(a - 5, b + 5, len_test_dataset).reshape(-1, 1)
    y_plot = np.full((dataset_size_plot, 1), np.nan)

    pre_processer_x = StandardScaler()
    pre_processer_x.fit(x)

    pre_processer_y = StandardScaler()
    pre_processer_y.fit(y)

    if generate_ood:
        x_train = np.linspace(x_plot.min(), x_plot.max(), len_train_dataset).reshape(-1, 1)
        y_train = np.full((len_train_dataset, 1), np.nan)

        x_test = np.linspace(x_plot.min(), x_plot.max(), len_test_dataset).reshape(-1, 1)
        y_test = np.full((len_test_dataset, 1), np.nan)

    scale = True
    x_train, y_train = post_process_data(pre_processer_x, pre_processer_y, x_train, y_train, scale=scale, **kwargs)
    x_test, y_test = post_process_data(pre_processer_x, pre_processer_y, x_test, y_test, scale=scale, **kwargs)
    x_plot, y_plot = post_process_data(pre_processer_x, pre_processer_y, x_plot, y_plot, scale=scale, **kwargs)

    input_size = x_train.shape[-1]
    output_size = y_train.shape[-1]

    return x_train, y_train, x_test, y_test, x_plot, y_plot, input_size, output_size

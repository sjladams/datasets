from typing import Optional
import os
import torch
import numpy as np

import datasets.core.utils as utils
from datasets.core.templates import RegressionDataset
import datasets.core.transformers as tf


def gp_sample(x: torch.Tensor, ampl: float=1., leng: float=1., sn2: float=0.1):
    n, x = x.shape[0], x / leng
    sum_xx = torch.sum(x * x, dim=1).view(-1, 1).repeat(1, n)
    D = sum_xx + sum_xx.t() - 2 * torch.matmul(x, x.t())
    C = ampl**2 * torch.exp(-0.5 * D) + torch.eye(n) * sn2
    return torch.distributions.MultivariateNormal(torch.zeros(n), C).sample().view(-1, 1)


def load_gp_samples(
        dataset_name: str,
        train: bool = True,
        len_dataset: Optional[int] = 512,
        ood: bool = False,
        gap: Optional[list] = None, # [-0.6, 0.1]
        **kwargs):
    """
    gp_samples was previously named example1d, regr1d and synthetic1d
    """
    data_path = f"{utils.get_package_data_root(dataset_name)}{os.sep}{'train' if train else 'test'}_data.csv"

    if os.path.exists(data_path):
        data = np.loadtxt(data_path, delimiter=',', skiprows=1)
        data = torch.from_numpy(data).to(torch.float32)
        x = data[..., 0].view(-1, 1)
        y = data[..., 1].view(-1, 1)
    else:
        raise FileNotFoundError(f"Data file not found: {data_path}")
        x = utils.rand((len_dataset, 1), l=-10., u=10.)
        y = gp_sample(x, ampl=1.6, leng=1.8)

    if gap is not None:
        mask = torch.logical_and(x > gap[0], x < gap[1])
        x = x[~mask].view(-1, 1)
        y = y[~mask].view(-1, 1)

    if ood:
        raise NotImplementedError
        # ood_domain_exp_factor: float = 0.25
        # domain = [inputs.min(), inputs.max()]
        # ood_expansion = (domain[1] - domain[0]) * ood_domain_exp_factor
        # inputs = torch.linspace(domain[0] - ood_expansion, domain[1] + ood_expansion, len_dataset).view(-1, 1)
        # outputs = torch.zeros(len_dataset).fill_(torch.nan).view(-1, 1)

    x = x[:len_dataset]
    y = y[:len_dataset]

    return RegressionDataset(
        data=x,
        targets=y,
        train=train,
        ood=ood,
        transform=NormalizeTransform(mean=x.mean(0), std=x.std(0)),
        target_transform=NormalizeTransform(mean=y.mean(0), std=y.std(0))
    )


def load_snelson(
        dataset_name: str,
        train: bool = True,
        len_dataset: Optional[int] = 512,
        ood: bool = False,
        gap: bool = False,
        **kwargs):

    data_path = f"{utils.get_package_data_root(dataset_name)}{os.sep}data.txt.gz"

    if os.path.exists(data_path):
        data = utils.open_txt_gz(data_path, dtype=torch.float32)
        x, y = data[..., 0], data[..., 1]
    else:
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if gap: # as in https://arxiv.org/abs/1906.01930 remove all points between 3 and 5:
        mask = torch.logical_and(x > 3., x < 5.)
        x = x[~mask].view(-1, 1)
        y = y[~mask].view(-1, 1)

    if ood:
        raise NotImplementedError
        # x_train = np.linspace(x.min() - ood_expansion, x.max() + ood_expansion, len_train_dataset)[..., None]
        # y_train = np.full((len_train_dataset, 1), np.nan)
        #
        # x_test = np.linspace(x.min() - ood_expansion, x.max() + ood_expansion, len_test_dataset)[..., None]
        # y_test = np.full((len_test_dataset, 1), np.nan)

    if len_dataset < len(x):
        indices = utils.generate_train_test_set_indices(dataset_name, train, len_dataset, len(x))
        x, y = x[indices], y[indices]

    return RegressionDataset(
        data=x,
        targets=y,
        train=train,
        ood=ood,
        transform=tf.NormalizeNumerical(mean=x.mean(0), std=x.std(0)),
        target_transform=tf.NormalizeNumerical(mean=y.mean(0), std=y.std(0))
    )

mapper = {
    'gp_samples': load_gp_samples,
    'snelson': load_snelson
}

def load_other_regression(dataset_name: str, **kwargs):
    return mapper[dataset_name](dataset_name, **kwargs)
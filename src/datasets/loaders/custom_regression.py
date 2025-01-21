from typing import Optional
import os
import torch

import datasets.core.utils as utils
from datasets.core.templates import RegressionDataset
import datasets.core.transformers as tf


def linear(len_dataset: int, in_features: int, out_features: int):
    x = utils.rand((len_dataset, in_features), -1, 1)
    y = torch.nn.functional.linear(x, torch.ones(in_features, out_features) * 0.5) + torch.randn(
        (len_dataset, out_features)) * 0.1
    return x, y


def noisy_sine(len_dataset: int, in_features: int, out_features: int):
    """
    Generates a noisy sine dataset. Includes old datasets noisysine1 and noisysine2d (different implementation)
    """
    if in_features != out_features:
        raise NotImplementedError

    x = utils.rand((len_dataset, in_features), -1, 1)
    y = torch.sin(2 * torch.pi * x) + torch.randn((len_dataset, out_features)) * (1. - torch.sin(2 * torch.pi * x).abs())
    return x, y


def _dynamics_uva_tutorial(x: torch.Tensor):
    return x + 0.3 * torch.sin(2 * torch.pi * x) + 0.3 * torch.sin(4 * torch.pi * x)


def uva_tutorial(len_dataset: int, in_features: int, out_features: int):
    """
    Dataset used in tutorial: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Bayesian_Neural_Networks/dl2_bnn_tut1_students_with_answers.html
    """
    if in_features != 1 or out_features != 1:
        raise NotImplementedError

    x = torch.cat(
        (torch.linspace(-0.2, 0.2, len_dataset // 2),
         torch.linspace(0.6, 1, len_dataset // 2))
    ).view(-1, 1)

    noise = 0.02 * torch.randn(x.shape)
    y = _dynamics_uva_tutorial(x + noise)
    return x, y


data_generating_mapper = {
    "linear": linear,
    "noisy_sine": noisy_sine,
    "uva_tutorial": uva_tutorial
}


def load_custom_regression(
        dataset_name: str,
        train: bool = True,
        len_dataset: Optional[int] = 2**10,
        ood: bool = False,
        in_features: int = 1,
        out_features: int = 1,
        **kwargs):

    data_path = f"{utils.get_local_data_root(dataset_name)}{os.sep}{'train' if train else 'test'}_data_in={in_features}_out={out_features}_size={len_dataset}.txt.gz"
    if os.path.exists(data_path):
        data = utils.open_txt_gz(data_path, dtype=torch.float32)

        x = data[... , range(in_features)]
        y = data[...,  range(-out_features, 0)]
    else:
        x, y = data_generating_mapper[dataset_name](len_dataset, in_features, out_features)
        utils.save_txt_gz(data_path, torch.cat((x, y), dim=-1))

    if ood:
        raise NotImplementedError

    return RegressionDataset(
        data=x,
        targets=y,
        train=train,
        ood=ood,
        transform=tf.NormalizeNumerical(mean=x.mean(0), std=x.std(0)),
        target_transform=tf.NormalizeNumerical(mean=y.mean(0), std=y.std(0))
    )

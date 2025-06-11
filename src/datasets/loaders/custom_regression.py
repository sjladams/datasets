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


def noisy_sine(len_dataset: int, in_features: int, out_features: int, variance: float = 0.05 ** 2):
    """
    Generates a noisy sine dataset. Includes old datasets noisysine1 and noisysine2d (different implementation)
    """
    if in_features != out_features:
        raise NotImplementedError

    x = utils.rand((len_dataset, in_features), -1, 1)
    y = torch.sin(2 * torch.pi * x) + torch.randn((len_dataset, out_features)) * variance 
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


def _load_or_generate_data(
        dataset_name: str,
        train: bool,
        len_dataset: int,
        in_features: int,
        out_features: int,
):
    data_path = f"{utils.get_local_data_root(dataset_name)}{os.sep}{'train' if train else 'test'}_data_in={in_features}_out={out_features}_size={len_dataset}.txt.gz"
    if os.path.exists(data_path):
        data = utils.open_txt_gz(data_path, dtype=torch.float32)

        x = data[... , range(in_features)]
        y = data[...,  range(-out_features, 0)]
    else:
        # test set is 10% of the training set
        if not train:
            len_dataset = int(0.1 * len_dataset)

        x, y = data_generating_mapper[dataset_name](len_dataset, in_features, out_features)
        utils.save_txt_gz(data_path, torch.cat((x, y), dim=-1))
    return x, y


def load_custom_regression(
        dataset_name: str,
        train: bool = True,
        len_dataset: int = 2**10,
        ood: bool = False,
        in_features: int = 1,
        out_features: int = 1,
        **kwargs):
    """
    Notes:
    - len_dataset is the number of samples in the train dataset
    """
    x, y = _load_or_generate_data(dataset_name, train=train, len_dataset=len_dataset, in_features=in_features, out_features=out_features)

    if ood:
        raise NotImplementedError

    # Use training data to normalize the data
    if train or not ood:
        transform = tf.NormalizeNumerical(mean=x.mean(0), std=x.std(0))
        target_transform = tf.NormalizeNumerical(mean=y.mean(0), std=y.std(0))
    else:
        x_train, y_train = _load_or_generate_data(dataset_name, train=False, len_dataset=len_dataset, in_features=in_features, out_features=out_features)
        transform = tf.NormalizeNumerical(mean=x_train.mean(0), std=x_train.std(0))
        target_transform = tf.NormalizeNumerical(mean=y_train.mean(0), std=y_train.std(0))

    return RegressionDataset(
        data=x,
        targets=y,
        name=dataset_name,
        train=train,
        ood=ood,
        transform=transform,
        target_transform=target_transform
    )

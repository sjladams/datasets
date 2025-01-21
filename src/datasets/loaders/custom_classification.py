import os
from typing import Optional
import torch
from sklearn.datasets import make_moons

from datasets.core.transformers import FlattenTransform, NormalizeTransform, Compose, FloatTransform, EnsureChannelTransform
from datasets.core.templates import ClassificationDataset
import datasets.core.utils as utils


def half_moons(len_dataset: int, in_features: int=2, noise=0.1):
    if in_features != 2:
        raise ValueError

    data, targets = make_moons(n_samples=len_dataset, noise=noise)
    return torch.as_tensor(data, dtype=torch.float32), torch.as_tensor(targets, dtype=torch.int64)


def vertical_split(len_dataset: int, in_features: int=2):
    if in_features != 2:
        raise ValueError

    # Generate random points in the range [0, 1] for both dimensions
    data = torch.rand(len_dataset, 2)

    # Scale points to the range [-1, 1] for both dimensions
    data = data * 2 - 1

    # Shift x-coordinates to the range [-1, 0] for class 0 and to range [0, 1] for class 1
    data[:len_dataset // 2, 0] = data[:len_dataset // 2, 0] / 2 - 0.5
    data[len_dataset // 2:, 0] = data[len_dataset // 2:, 0] / 2 + 0.5

    targets = torch.cat([torch.zeros(len_dataset // 2, dtype=torch.int64),
                         torch.ones(len_dataset // 2, dtype=torch.int64)])
    return data, targets


def diagonal_split(len_dataset: int, in_features: int=2):
    if in_features != 2:
        raise ValueError

    # Generate random points in the range [-1, 1] for both dimensions
    data = torch.rand(len_dataset, 2) * 2 - 1

    # Assign class 0 to points below the diagonal y = x and class 1 to points above the diagonal
    targets = (data[:, 1] > -data[:, 0]).long()

    return data, targets


def ellipsoid_split(len_dataset: int, in_features: int=2, epsilon_x: float = 1.0, epsilon_y: float = 4.0):
    if in_features != 2:
        raise ValueError

    # Generate random points in the range [-1, 1] for both dimensions
    data = torch.rand(len_dataset, 2) * 2 - 1

    # Assign class 0 to points within the epsilon radius and class 1 to all other points
    targets = (data[:, 0].pow(2) * epsilon_x + data[:, 1].pow(2) * epsilon_y <= 0.8).long()

    return data, targets


data_generating_mapper = {
    "half_moons": half_moons,
    "vertical_split": vertical_split,
    "diagonal_split": diagonal_split,
    "ellipsoid_split": ellipsoid_split
}


def load_custom_classification(
        dataset_name: str,
        train: bool = True,
        len_dataset: Optional[int] = 2 ** 10,
        ood: bool = False,
        in_features: int = 2,
        **kwargs):

    basis_path = f"{utils.get_local_data_root(dataset_name)}{os.sep}{'train' if train else 'test'}_in={in_features}_size={len_dataset}"
    data_path = f"{basis_path}_data.txt.gz"
    targets_path = f"{basis_path}_targets.txt.gz"
    if os.path.exists(data_path) and os.path.exists(targets_path):
        data = utils.open_txt_gz(data_path, dtype=torch.float32)
        targets = utils.open_txt_gz(data_path, dtype=torch.int64)
    else:
        data, targets = data_generating_mapper[dataset_name](len_dataset=len_dataset, in_features=in_features)
        # ensure equal number of points per class:
        data, targets = utils.balance_classification_data(data, targets)

        utils.save_txt_gz(data_path, data)
        utils.save_txt_gz(targets_path, targets)

    if ood:
        raise NotImplementedError

    return ClassificationDataset(
        data=data,
        targets=targets,
        train=train,
        image_mode=f"POINTS_{in_features}d",
        transform=NormalizeTransform(mean=data.mean(0), std=data.std(0)),
    )

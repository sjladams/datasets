from typing import Union

from datasets.core.templates import ClassificationDataset, RegressionDataset
from datasets.loaders import load_custom_regression, load_torchvision, load_uci, load_other_regression, load_medmnist, load_custom_classification

name_mapping = {
    'mnist': load_torchvision,
    'fashion_mnist': load_torchvision,
    'cifar10': load_torchvision,
    'cifar100': load_torchvision,
    'kin40k': load_uci,
    'energy': load_uci,
    'houseelectric': load_uci,
    'protein': load_uci,
    'wine': load_uci,
    'noisy_sine': load_custom_regression,
    'linear': load_custom_regression,
    'uva_tutorial': load_custom_regression,
    'gp_samples': load_other_regression,
    'snelson': load_other_regression,
    'path_mnist': load_medmnist,
    'oct_mnist': load_medmnist,
    "half_moons": load_custom_classification,
    "vertical_split": load_custom_classification,
    "diagonal_split": load_custom_classification,
    "ellipsoid_split": load_custom_classification
}


def get_dataset(dataset_name: str, **kwargs) -> Union[RegressionDataset, ClassificationDataset]:
    data_loader = _get_dataset_loader(dataset_name)
    return data_loader(dataset_name=dataset_name, **kwargs)


def _get_dataset_loader(dataset_nane: str):
    if not dataset_nane in name_mapping:
        raise NotImplementedError(f"Dataset {dataset_nane} not implemented.")
    else:
        return name_mapping[dataset_nane]


class Info:
    datasets = list(name_mapping.keys())

info = Info()
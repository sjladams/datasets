from torch.utils.data import DataLoader, Dataset
import torch
from typing import Tuple
import numpy as np

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
from .synthetic1d import load_synthetic1d
from .prior1d import load_prior1d


def get_dataset(dataset_name: str, paths=False, **kwargs) -> Tuple[Dataset, torch.Size, torch.Size]:
    data_loader = _get_dataset_loader(dataset_name)
    ds, input_size, output_size = data_loader(paths=paths, dataset_name=dataset_name, **kwargs)
    return ds, input_size, output_size


def get_data_loader(dataset_name: str, batch_size: int, shuffle: bool = False, paths=False, **kwargs) -> \
        Tuple[DataLoader, torch.Size, torch.Size]:
    ds, input_size, output_size = get_dataset(dataset_name=dataset_name, paths=paths, **kwargs)
    loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle,
                        worker_init_fn=np.random.seed(0), num_workers=0)
    return loader, input_size, output_size


def _get_dataset_loader(dataset_name: str):
    if dataset_name == 'noisysine1d':
        raise NotImplementedError
        return load_noisysine1d
    elif dataset_name == 'noisysine2d':
        raise NotImplementedError
        return load_noisysine2d
    elif dataset_name == 'linear1d':
        raise NotImplementedError
        return load_linear
    elif dataset_name == 'kin8nm':
        return load_kin8nm
    elif dataset_name == "regr1d":
        raise NotImplementedError
        return load_regr1d
    elif dataset_name == 'mnist':
        return load_mnist
    elif dataset_name == 'fashion_mnist':
        return load_fashion_mnist
    elif dataset_name == 'snelson1d':
        raise NotImplementedError
        return load_snelson1d
    elif dataset_name == 'boston':
        return load_boston
    elif dataset_name == 'concrete':
        return load_concrete
    elif dataset_name == 'energy':
        return load_energy
    elif dataset_name == 'naval':
        raise NotImplementedError
        return load_naval
    elif dataset_name == 'power':
        raise NotImplementedError
        return load_power
    elif dataset_name == 'protein':
        raise NotImplementedError
        return load_protein
    elif dataset_name == 'wine':
        raise NotImplementedError
        return load_wine
    elif dataset_name == 'luca':
        raise NotImplementedError
        return load_luca
    elif dataset_name == 'luca2d':
        raise NotImplementedError
        return load_luca2d
    elif dataset_name == 'cifar10':
        return load_cifar10
    elif dataset_name == 'example1d':
        return load_example1d
    elif dataset_name == 'uva_tutorial':
        raise NotImplementedError
        return load_uva_tutorial
    elif dataset_name == 'synthetic1d':
        raise NotImplementedError
        return load_synthetic1d
    elif dataset_name == 'prior1d':
        return load_prior1d
    else:
        raise AssertionError("\nDataset not available.")

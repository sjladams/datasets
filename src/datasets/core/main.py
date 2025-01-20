from torch.utils.data import Dataset

# from src.datasets.loaders.regr1d import load_regr1d
from datasets.loaders import load_custom_regression, load_torchvision, load_uci, load_other_regression

# from src.datasets.loaders.example1d import load_example1d
# from src.datasets.loaders.uva_tutorial import load_uva_tutorial
# from src.datasets.loaders.prior1d import load_prior1d


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
}


def get_dataset(dataset_name: str, **kwargs) -> Dataset:
    data_loader = _get_dataset_loader(dataset_name)
    return data_loader(dataset_name=dataset_name, **kwargs)


def _get_dataset_loader(dataset_nane: str):
    if not dataset_nane in name_mapping:
        raise NotImplementedError(f"Dataset {dataset_nane} not implemented.")
    else:
        return name_mapping[dataset_nane]
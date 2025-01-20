from torchvision import datasets
from typing import Optional
import torch

from datasets.core.transformers import FlattenTransform, NormalizeTransform, Compose, FloatTransform, EnsureChannelTransform
from datasets.core.templates import ClassificationDataset
from datasets.core.utils import get_local_data_root

mapper = {
    "mnist":
        {"image_mode": "L",
         "dataset": datasets.MNIST,
         "train_len": 60000,
         "test_len": 10000
         },
    "fashion_mnist":
        {"image_mode": "L",
         "dataset": datasets.FashionMNIST,
         "train_len": 60000,
         "test_len": 10000
         },
    "cifar10":
        {"image_mode": "RGB",
         "dataset": datasets.CIFAR10,
         "train_len": 50000,
         "test_len": 10000
         },
    "cifar100":
        {"image_mode": "RGB",
         "dataset": datasets.CIFAR100,
         "train_len": 50000,
         "test_len": 10000
         }
}

def load_torchvision(
        dataset_name: str,
        train: bool = True,
        len_dataset: Optional[int] = None,
        flatten: bool = True,
        **kwargs) -> ClassificationDataset:

    if dataset_name not in mapper:
        raise NotImplementedError(f"Torchvision Dataset {dataset_name} not implemented.")
    else:
        ds = mapper[dataset_name]['dataset'](get_local_data_root(dataset_name), train=train, download=True)

    transformer = [
        EnsureChannelTransform(ds[0][0].size),
        FloatTransform(),
        NormalizeTransform(mean=0., std=torch.iinfo(ds.data.dtype).max)
    ]

    if flatten:
        transformer += [FlattenTransform(mapper[dataset_name]['image_mode'])]
    else:
        transformer = None

    if len_dataset is None:
        len_dataset = len(ds)

    return ClassificationDataset(
        data=ds.data[:len_dataset],
        targets=ds.targets[:len_dataset],
        train=train,
        transform=Compose(transformer),
        image_mode=mapper[dataset_name]['image_mode']
    )

from torchvision import datasets
from typing import Optional
import torch

import datasets.core.transformers as tf
from datasets.core.templates import ClassificationDataset
from datasets.core.utils import get_local_data_root

mapper = {
    "mnist":
        {"image_mode": "L",
         "layout": "HW",
         "dataset": datasets.MNIST,
         "train_len": 60000,
         "test_len": 10000
         },
    "fashion_mnist":
        {"image_mode": "L",
         "layout": "HW",
         "dataset": datasets.FashionMNIST,
         "train_len": 60000,
         "test_len": 10000
         },
    "cifar10":
        {"image_mode": "RGB",
         "layout": "HWC",
         "dataset": datasets.CIFAR10,
         "train_len": 50000,
         "test_len": 10000
         },
    "cifar100":
        {"image_mode": "RGB",
         "layout": "HWC",
         "dataset": datasets.CIFAR100,
         "train_len": 50000,
         "test_len": 10000
         }
}

def load_torchvision(
        dataset_name: str,
        train: bool = True,
        len_dataset: Optional[int] = None,
        flatten: bool = False,
        **kwargs) -> ClassificationDataset:

    if dataset_name not in mapper:
        raise NotImplementedError(f"Torchvision Dataset {dataset_name} not implemented.")
    else:
        ds = mapper[dataset_name]['dataset'](get_local_data_root(dataset_name), train=train, download=True)
        data, targets = torch.as_tensor(ds.data).type(torch.uint8), torch.as_tensor(ds.targets).type(torch.int64)

    transformer = [
        tf.ToCHW(mapper[dataset_name]['layout']),
        tf.Float(),
        tf.NormalizeImage(mean=0., std=torch.iinfo(data.dtype).max)
    ]

    if flatten:
        transformer += [tf.Flatten(mapper[dataset_name]['image_mode'])]

    if len_dataset is None:
        len_dataset = len(data)

    return ClassificationDataset(
        data=data[:len_dataset],
        targets=targets[:len_dataset],
        name=dataset_name,
        train=train,
        transform=tf.Compose(transformer),
        image_mode=mapper[dataset_name]['image_mode']
    )

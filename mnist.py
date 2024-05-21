from torchvision import datasets
import os
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from typing import Tuple

from .utils import FlattenTransform, ClassificationDataset, points_to_paths


def load_mnist(train: bool = True, len_dataset: int = 60000, flatten: bool = True, paths: bool = False, **kwargs) -> \
        Tuple[Dataset, torch.Size, torch.Size]:
    root = '{}/{}'.format(os.path.dirname(__file__), '/mnist')
    dataset = datasets.MNIST(root, train=train, download=True)
    num_classes = len(dataset.classes)

    if flatten:
        transformer = FlattenTransform(start_dim=-2)
    else:
        transformer = None

    dataset = ClassificationDataset(train=train, paths=paths, data=dataset.data, targets=dataset.targets,
                                    transform=transformer, len_dataset=len_dataset, mode="L", num_classes=num_classes,
                                    **kwargs)

    input_size = dataset[0][0].shape
    output_size = dataset[0][1].shape

    return dataset, input_size, output_size

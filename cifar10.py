from torchvision import datasets
import os
from torchvision import transforms
import torch
from typing import Tuple
from torch.utils.data import Dataset

from .utils import FlattenTransform, MoveChannelTransform, ClassificationDataset


def load_cifar10(train: bool = True, len_dataset: int = 50000, flatten: bool = False, paths: bool = False, **kwargs) \
        -> Tuple[Dataset, torch.Size, torch.Size]:
    # # create grid of path_length * classes for plotting
    # mask = []
    # path_length_per_class = kwargs['path_length'] // 10
    # for idx in range(10):
    #     mask.append(torch.where(y_test[:, idx] == 1)[0][:path_length_per_class])
    # mask = torch.cat(mask)
    # x_plot = x_test[mask]
    # y_plot = y_test[mask]

    root = '{}/{}'.format(os.path.dirname(__file__), '/cifar10')
    dataset = datasets.CIFAR10(root, train=train, download=True)
    num_classes = len(dataset.classes)

    if flatten:
        transformer = FlattenTransform(start_dim=-3)
    else:
        transformer = None

    dataset = ClassificationDataset(train=train, paths=paths, data=torch.from_numpy(dataset.data),
                                    targets=dataset.targets, transform=transformer, len_dataset=len_dataset, mode=None,
                                    num_classes=num_classes, **kwargs)

    input_size = dataset[0][0].shape
    output_size = dataset[0][1].shape

    return dataset, input_size, output_size


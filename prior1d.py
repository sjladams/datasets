import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch
from typing import Optional, Callable, List, Any, Tuple

from .utils import points_to_paths, NormalizeTransform


class Prior1dDataset(Dataset):
    def __init__(self, input_transform: Optional[Callable] = None, output_transform: Optional[Callable] = None,
                 paths: bool = False, **kwargs):
        self.paths = paths
        self.input_transform, self.output_transform = input_transform, output_transform
        self.inputs, self.outputs = self._load_data(**kwargs)

    def _load_data(self, len_dataset: Optional[int] = None, **kwargs):
        if len_dataset is None:
            len_dataset = 1000
        inputs = torch.linspace(-1, 1, len_dataset).unsqueeze(-1)
        outputs = torch.ones(inputs.shape).fill_(torch.nan)

        if self.paths:
            inputs, outputs = points_to_paths(x=inputs, y=outputs, **kwargs)

        return inputs, outputs

    @property
    def mean_inputs(self):
        return self.inputs.mean(0)

    @property
    def std_inputs(self):
        return self.inputs.std(0)

    @property
    def mean_outputs(self):
        return self.outputs.mean(0)

    @property
    def std_outputs(self):
        return self.outputs.std(0)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        input, output = self.inputs[index], self.outputs[index]

        if self.input_transform is not None:
            input = self.input_transform(input)

        if self.output_transform is not None:
            output = self.output_transform(output)

        return input, output


def load_prior1d(train: bool = True, len_dataset: int = 512, ood: bool = False, paths: bool = False, **kwargs) -> \
        Tuple[Dataset, torch.Size, torch.Size]:
    train_dataset_no_gap = Prior1dDataset()
    input_transform = NormalizeTransform(mean=train_dataset_no_gap.mean_inputs, std=train_dataset_no_gap.std_inputs)
    output_transform = NormalizeTransform(mean=train_dataset_no_gap.mean_outputs, std=train_dataset_no_gap.std_outputs)

    dataset = Prior1dDataset(paths=paths, train=train, ood=ood, len_dataset=len_dataset,
                             input_transform=input_transform, output_transform=output_transform, **kwargs)
    input_size = dataset[0][0].shape
    output_size = dataset[0][1].shape

    return dataset, input_size, output_size

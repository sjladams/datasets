import os
import numpy as np
from torch.utils.data import Dataset
import torch
from typing import Optional, Callable, List, Any, Tuple

from .utils import points_to_paths, NormalizeTransform, RegressionDataset


class Example1dDataset(RegressionDataset):
    def __init__(self, **kwargs):
        super(Example1dDataset, self).__init__(**kwargs)

    def _load_data(self, len_dataset: Optional[int] = None, gap: Optional[list] = None,
                   ood_domain_exp_factor: float = 0.25, **kwargs):
        file = f"{'train' if self.train else 'test'}_data.csv"
        data = np.loadtxt(os.path.join(self.root, file), delimiter=',', skiprows=1)
        data = torch.from_numpy(data).to(torch.float32)
        inputs, outputs = data[:, 0], data[:, 1]

        if gap is not None:
            mask = torch.logical_and(inputs > gap[0], inputs < gap[1])
            inputs = inputs[~mask].reshape((-1, 1))
            outputs = outputs[~mask].reshape((-1, 1))

        if self.ood:
            if len_dataset is None:
                len_dataset = inputs.shape[0]

            domain = [inputs.min(), inputs.max()]
            ood_expansion = (domain[1] - domain[0]) * ood_domain_exp_factor
            inputs = torch.linspace(domain[0] - ood_expansion, domain[1] + ood_expansion, len_dataset).view(-1, 1)
            outputs = torch.zeros(len_dataset).fill_(torch.nan).view(-1, 1)
        elif len_dataset is not None:
            inputs = inputs[:len_dataset]
            outputs = outputs[:len_dataset]

        if self.paths:
            inputs, outputs = points_to_paths(x=inputs, y=outputs, **kwargs)

        return inputs, outputs


def load_example1d(train: bool = True, len_dataset: int = 512, ood: bool = False, paths: bool = False,
                   data_specs: dict = None, **kwargs) -> Tuple[Dataset, torch.Size, torch.Size]:
    if data_specs is None:
        data_specs = dict()

    root = '{}/{}'.format(os.path.dirname(__file__), '/example1d/')
    train_dataset_no_gap = Example1dDataset(root=root, train=True)
    input_transform = NormalizeTransform(mean=torch.zeros(1), std=train_dataset_no_gap.std_inputs)
    output_transform = NormalizeTransform(mean=train_dataset_no_gap.mean_outputs, std=train_dataset_no_gap.std_outputs)

    dataset = Example1dDataset(root=root, paths=paths, train=train, ood=ood, len_dataset=len_dataset,
                               input_transform=input_transform, output_transform=output_transform, **data_specs,
                               **kwargs)
    input_size = dataset[0][0].shape
    output_size = dataset[0][1].shape

    return dataset, input_size, output_size

import math
from typing import Optional, Callable, List, Any, Tuple
import torch
import os
import numpy as np

from .utils import points_to_paths, NormalizeTransform, RegressionDataset


def f(x: torch.Tensor):
    return x + 0.3 * torch.sin(2 * math.pi * x) + 0.3 * torch.sin(4 * math.pi * x)


class UVATutorialDataset(RegressionDataset):
    def __init__(self, **kwargs):
        super(UVATutorialDataset, self).__init__(**kwargs)

    def _load_data(self, len_dataset: Optional[int] = 1000, ood_domain_exp_factor: float = 0.25, **kwargs):
        if self.ood:
            inputs = torch.linspace(-0.5, 1.5, len_dataset).unsqueeze(-1)
            outputs = f(inputs)
        else:
            data_root = f"{self.root}{os.sep}data.txt.gz"
            if os.path.exists(data_root):
                data = np.loadtxt(data_root)
                data = torch.from_numpy(data).to(torch.float32)
                inputs, outputs = data[:, :-1], data[:, -1:]
            else:
                inputs = torch.cat([torch.linspace(-0.2, 0.2, len_dataset // 2),
                                    torch.linspace(0.6, 1, len_dataset // 2)]).unsqueeze(-1)
                noise = 0.02 * torch.randn(inputs.shape)  # sigma 0.2
                outputs = f(inputs + noise)
                np.savetxt(data_root, np.concatenate((inputs, outputs), axis=1))

        if len_dataset is None:
            len_dataset = inputs.shape[0]
        else:
            len_dataset = min(inputs.shape[0], len_dataset)

        if self.train:
            inputs, outputs = inputs[:len_dataset], outputs[:len_dataset]
        else:
            inputs, outputs = inputs[-len_dataset:], outputs[-len_dataset:]

        if self.paths:
            inputs, outputs = points_to_paths(x=inputs, y=outputs, **kwargs)

        return inputs, outputs


def load_uva_tutorial(train: bool = True, len_dataset: int = 1000, ood: bool = False, paths: bool = False, **kwargs):
    root = os.path.join(os.path.dirname(__file__), 'uva_tutorial')

    # train_dataset = UVATutorialDataset(root=root, len_dataset=1000, train=True, ood=False)
    # input_transform = NormalizeTransform(mean=train_dataset.mean_inputs, std=train_dataset.std_inputs)
    # output_transform = NormalizeTransform(mean=train_dataset.mean_outputs, std=train_dataset.std_outputs)

    dataset = UVATutorialDataset(root=root, paths=paths, train=train, ood=ood, len_dataset=len_dataset, **kwargs)

    input_size = dataset[0][0].shape
    output_size = dataset[0][1].shape

    return dataset, input_size, output_size

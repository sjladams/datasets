from typing import Optional
import os
import torch

from datasets.core.templates import RegressionDataset
from datasets.core.utils import get_package_data_root, open_txt_gz, save_txt_gz, get_local_data_root, generate_train_test_set_indices #\todo import utils as utils
from datasets.core.transformers import NormalizeTransform

mapper = {
    "kin8nm":
        {"idx_x": range(8),
         "idx_y": [8],
         "len": 8192
         },
    "concrete":
        {"idx_x": range(8),
         "idx_y": [8],
         "len": 1030
         },
    "naval":
        {"idx_x": range(16),
         "idx_y": [16, 17],
         "len": 11934
         },
    "power":
        {"idx_x": range(4),
         "idx_y": [4],
         "len": 9568
         },
    "boston":
        {"idx_x": range(12),
         "idx_y": [12],
         "len": 506
         },
    "protein":
        {"idx_x": range(9),
         "idx_y": [9],
         "len": 45730
         },
    "wine":
        {"idx_x": range(11),
         "idx_y": [11],
         "len": 1599
         },
    "energy":
        {"idx_x": range(8),
         "idx_y": [8],
         "len": 768
         }
}

def load_uci(
        dataset_name: str,
        train: bool = True,
        len_dataset: Optional[int] = None,
        ood: bool = False,
        **kwargs) -> RegressionDataset:

    if dataset_name not in mapper:
        raise NotImplementedError(f"UCI Dataset {dataset_name} not implemented.")
    else:
        data_path = f"{get_package_data_root(dataset_name)}{os.sep}data.txt.gz"
        if os.path.exists(data_path):
            data = open_txt_gz(data_path, dtype=torch.float32)

            y = data[... , mapper[dataset_name]['idx_y']]
            x = data[...,  mapper[dataset_name]['idx_x']]
            assert y.shape[-1] + x.shape[-1] <= data.shape[-1]
        else:
            raise NotImplementedError('data file not available')

    if len_dataset < len(x):
        indices = generate_train_test_set_indices(dataset_name, train, len_dataset, len(x))
        x, y = x[indices], y[indices]

    if ood:
        raise NotImplementedError('OOD sampling not implemented')
        ratio_ood_samples = 0.5
        uniform_ood_sampling = False
        uniform_ood_sapling_dim = 0
        #
        #     num_ood_samples = int(inputs.shape[0] * ratio_ood_samples / (1 - ratio_ood_samples))
        #     if uniform_ood_sampling:
        #         # set all dimensions other than uniform_ood_sapling_dim to means
        #         domain_sampling_dim = [inputs[:, uniform_ood_sapling_dim].min(),
        #                                inputs[:, uniform_ood_sapling_dim].max()]
        #         inputs = inputs.mean(0)
        #         inputs = inputs.unsqueeze(0).expand(len_dataset, in_features).clone()
        #         inputs[:, uniform_ood_sapling_dim] = torch.linspace(*domain_sampling_dim, len_dataset)
        #         outputs = torch.zeros(len_dataset, out_features).fill_(torch.nan)
        #     else:
        #         ood_indices = torch.randint(0, len_dataset, (num_ood_samples,))
        #         ood_inputs = inputs[ood_indices] + torch.randn((num_ood_samples, in_features)) * 0.05
        #         ood_outputs = torch.ones((num_ood_samples, out_features)).fill_(torch.nan)
        #
        #         # domain = [inputs.min(0).values, inputs.max(0).values]
        #         # ood_inputs = domain[0] + (domain[1] - domain[0]) * torch.rand(num_ood_samples, in_features)
        #         # ood_outputs = torch.zeros(num_ood_samples, out_features).fill_(torch.nan)
        #
        #         inputs = torch.cat((inputs, ood_inputs))
        #         outputs = torch.cat((outputs, ood_outputs))

    return RegressionDataset(
        data=x,
        targets=y,
        train=train,
        ood=ood,
        transform=NormalizeTransform(mean=x.mean(0), std=x.std(0)),
        target_transform=NormalizeTransform(mean=y.mean(0), std=y.std(0))
    )
from typing import Optional
import torch

from datasets.core.templates import RegressionDataset
import datasets.core.transformers as tf


def load_uci(
        dataset_name: str,
        train: bool = True,
        len_dataset: Optional[int] = None,
        split: int = 0,
        ood: bool = False,
        **kwargs) -> RegressionDataset:

    import uci_datasets  # python -m pip install git+https://github.com/treforevans/uci_datasets.git

    data = uci_datasets.Dataset(dataset_name)
    x_train, y_train, x_test, y_test = data.get_split(split=split)

    x_train, y_train = torch.from_numpy(x_train).type(torch.float32), torch.from_numpy(y_train).type(torch.float32)
    x_test, y_test = torch.from_numpy(x_test).type(torch.float32), torch.from_numpy(y_test).type(torch.float32)

    if train:
        x, y = x_train, y_train
    else:
        x, y = x_test, y_test

    if len_dataset is None:
        len_dataset = len(x)

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

    # use training data to normalize the data
    transform = tf.NormalizeNumerical(mean=x_train.mean(0), std=x_train.std(0))
    target_transform = tf.NormalizeNumerical(mean=y_train.mean(0), std=y_train.std(0))

    return RegressionDataset(
        data=x[:len_dataset],
        targets=y[:len_dataset],
        name=dataset_name,
        train=train,
        ood=ood,
        transform=transform,
        target_transform=target_transform
    )
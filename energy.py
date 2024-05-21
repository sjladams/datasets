from typing import Tuple

from .utils import UCIRegressionDataset, NormalizeTransform


def load_energy(train: bool = True, len_dataset=768, ood: bool = False, paths: bool = False,
                in_features: int = 8, out_features: int = 1, **kwargs) -> Tuple:
    train_dataset = UCIRegressionDataset(train=True, in_features=in_features, out_features=out_features, **kwargs)
    input_transform = NormalizeTransform(mean=train_dataset.mean_inputs, std=train_dataset.std_inputs)
    output_transform = NormalizeTransform(mean=train_dataset.mean_outputs, std=train_dataset.std_outputs)

    dataset = UCIRegressionDataset(train=train, paths=paths, ood=ood, len_dataset=len_dataset,
                                   in_features=in_features, out_features=out_features,
                                   input_transform=input_transform, output_transform=output_transform, **kwargs)
    input_size = dataset[0][0].shape
    output_size = dataset[0][1].shape

    return dataset, input_size, output_size
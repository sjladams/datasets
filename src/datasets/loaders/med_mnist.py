from typing import Optional
import torch

import datasets.core.transformers as tf
from datasets.core.templates import ClassificationDataset
from datasets.core.utils import get_local_data_root


def get_mapper():
    import medmnist

    return {
        "path_mnist":
            {"image_mode": "RGB",
             "layout": "HWC",
             "dataset": medmnist.PathMNIST,
             "train_len": 89996,
             "test_len": 10004
             },
        "oct_mnist":
            {"image_mode": "L",
             "layout": "HW",
             "dataset": medmnist.OCTMNIST,
             "train_len": 97477,
             "test_len": 10832
             },
    }


def load_medmnist(
        dataset_name: str,
        train: bool = True,
        len_dataset: Optional[int] = None,
        flatten: bool = False,
        **kwargs) -> ClassificationDataset:

    mapper = get_mapper()

    if dataset_name not in mapper:
        raise NotImplementedError(f"MedMNIST Dataset {dataset_name} not implemented.")
    else:
        ds = mapper[dataset_name]['dataset'](
            root=get_local_data_root(dataset_name),
            split='train' if train else 'val',
            download=True
        )

    transformer = [
        tf.ToCHW(mapper[dataset_name]['layout']),
        tf.Float(),
        tf.NormalizeImage(mean=0., std=255) # Note that in the example of the original code, the data is normalized between [-0.5, 0.5]
    ]

    if flatten:
        transformer += [tf.Flatten(mapper[dataset_name]['image_mode'])]

    if len_dataset is None:
        len_dataset = len(ds)

    return ClassificationDataset(
        data=torch.from_numpy(ds.imgs[:len_dataset]).type(torch.uint8),
        targets=torch.from_numpy(ds.labels[:len_dataset]).view(-1).type(torch.int64),
        name=dataset_name,
        train=train,
        transform=tf.Compose(transformer),
        image_mode=mapper[dataset_name]['image_mode']
    )

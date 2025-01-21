import pytest
import torch
from datasets import get_dataset


@pytest.mark.parametrize("dataset_name, flatten, data_size, target_size",
    [
    ("mnist", True, 784, 10),
    ("ellipsoid_split", False, 2, 1),
    ("kin40k", False, 8, 1),
    ])
def test_get_dataset(dataset_name, flatten, data_size, target_size):
    ds = get_dataset(dataset_name=dataset_name, flatten=flatten)
    assert ds.data_size == torch.Size((data_size,))
    assert ds.target_size == torch.Size((target_size,))


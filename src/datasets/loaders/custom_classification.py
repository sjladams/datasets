import os
from typing import Optional
import torch
from sklearn.datasets import make_moons

import datasets.core.transformers as tf
from datasets.core.templates import ClassificationDataset
import datasets.core.utils as utils


def get_projection_matrix(n: int, m: int, randomly: bool = True):
    """
    Generate a random projection matrix where columns are independent

    Args:
        n (int): Target dimensionality (number of rows in the matrix).
        m (int): Number of columns in the projection matrix.
        randomly (bool): If True, generate a random matrix. If False, generate a basis vector projection.
    """
    if randomly:
        # Generate a random matrix
        T = torch.randn(n, m)

        # Ensure orthogonality by QR decomposition
        T, _ = torch.linalg.qr(T)
    else:
        # Generate basis vector projection
        T = torch.zeros(n, m)
        indices = torch.randint(0, n, (m,))
        T[indices, :] = torch.eye(m)
    return T


def project_data(dataset_name: str, data: torch.Tensor, num_features: int, basis_features: int, randomly: bool):
    proj_mat_path = f"{utils.get_local_data_root(dataset_name)}{os.sep}proj_mat_in={num_features}{'_rand' if randomly else ''}.txt.gz"
    if os.path.exists(proj_mat_path):
        proj_mat = utils.open_txt_gz(proj_mat_path, dtype=torch.float32)
    else:
        proj_mat = get_projection_matrix(num_features, basis_features, randomly)
        utils.save_txt_gz(proj_mat_path, proj_mat)

    return data @ proj_mat.t()


def half_moons(len_dataset: int, num_features: int=2, noise=0.1, randomly: bool=False, **kwargs):
    data, targets = make_moons(n_samples=len_dataset, noise=noise)
    data, targets = torch.as_tensor(data, dtype=torch.float32), torch.as_tensor(targets, dtype=torch.int64)

    # if num_features > 2, project the data into an num_features-dimensional space
    if num_features == 2:
        pass
    elif num_features > 2:
        data = project_data("half_moons", data, num_features, 2, randomly)
    else:
        raise ValueError

    return data, targets


def vertical_split(len_dataset: int, num_features: int=2, randomly: bool=False, **kwargs):
    # Generate random points in the range [0, 1] for both dimensions
    data = torch.rand(len_dataset, 2)

    # Scale points to the range [-1, 1] for both dimensions
    data = data * 2 - 1

    # Shift x-coordinates to the range [-1, 0] for class 0 and to range [0, 1] for class 1
    data[:len_dataset // 2, 0] = data[:len_dataset // 2, 0] / 2 - 0.5
    data[len_dataset // 2:, 0] = data[len_dataset // 2:, 0] / 2 + 0.5

    targets = torch.cat([torch.zeros(len_dataset // 2, dtype=torch.int64),
                         torch.ones(len_dataset // 2, dtype=torch.int64)])

    # if num_features > 2, project the data into a num_features-dimensional space
    if num_features == 2:
        pass
    elif num_features > 2:
        data = project_data("vertical_split", data, num_features, 2, randomly)
    else:
        raise ValueError

    return data, targets


def diagonal_split(len_dataset: int, num_features: int=2, randomly: bool=False, **kwargs):
    # Generate random points in the range [-1, 1] for both dimensions
    data = torch.rand(len_dataset, 2) * 2 - 1

    # Assign class 0 to points below the diagonal y = x and class 1 to points above the diagonal
    targets = (data[:, 1] > -data[:, 0]).long()

    # if num_features > 2, project the data into an num_features-dimensional space
    if num_features == 2:
        pass
    elif num_features > 2:
        data = project_data("diagonal_split", data, num_features, 2, randomly)
    elif num_features < 2:
        raise ValueError

    return data, targets


def ellipsoid_split(len_dataset: int, num_features: int=2, epsilon_x: float = 1.0, epsilon_y: float = 4.0, randomly: bool=False, **kwargs):
    # Generate random points in the range [-1, 1] for both dimensions
    data = torch.rand(len_dataset, 2) * 2 - 1

    # Assign class 0 to points within the epsilon radius and class 1 to all other points
    targets = (data[:, 0].pow(2) * epsilon_x + data[:, 1].pow(2) * epsilon_y <= 0.8).long()

    # if num_features > 2, project the data into an num_features-dimensional space
    if num_features == 2:
        pass
    elif num_features > 2:
        data = project_data("ellipsoid_split", data, num_features, 2, randomly)
    else:
        raise ValueError

    return data, targets


data_generating_mapper = {
    "half_moons": half_moons,
    "vertical_split": vertical_split,
    "diagonal_split": diagonal_split,
    "ellipsoid_split": ellipsoid_split
}


def _load_or_generate_data(
        dataset_name: str,
        train: bool,
        len_dataset: Optional[int],
        num_features: int):
    basis_path = f"{utils.get_local_data_root(dataset_name)}{os.sep}{'train' if train else 'test'}_in={num_features}_size={len_dataset}"
    data_path = f"{basis_path}_data.txt.gz"
    targets_path = f"{basis_path}_targets.txt.gz"
    if os.path.exists(data_path) and os.path.exists(targets_path):
        data = utils.open_txt_gz(data_path, dtype=torch.float32)
        targets = utils.open_txt_gz(targets_path, dtype=torch.int64)
    else:
        # test set is 10% of the training set
        if not train:
            len_dataset = int(0.1 * len_dataset)

        data, targets = data_generating_mapper[dataset_name](len_dataset=len_dataset, num_features=num_features)

        # ensure equal number of points per class:
        data, targets = utils.balance_classification_data(data, targets)

        utils.save_txt_gz(data_path, data)
        utils.save_txt_gz(targets_path, targets)
    return data, targets


def load_custom_classification(
        dataset_name: str,
        train: bool = True,
        len_dataset: Optional[int] = 2 ** 10,
        ood: bool = False,
        num_features: int = 2,
        **kwargs):
    """
    Notes:
    - len_dataset is the number of samples in the train dataset
    """
    data, targets = _load_or_generate_data(dataset_name, train=train, len_dataset=len_dataset, num_features=num_features)

    if ood:
        data, targets = data_generating_mapper[dataset_name](len_dataset=len_dataset, num_features=num_features)

    # Use training data to normalize the data
    if train and not ood:
        transform = tf.NormalizeNumerical(mean=data.mean(0), std=data.std(0))
    else:
        train_data, _ = _load_or_generate_data(dataset_name, train=False, len_dataset=len_dataset, num_features=num_features)
        transform = tf.NormalizeNumerical(mean=train_data.mean(0), std=train_data.std(0))

    return ClassificationDataset(
        data=data,
        targets=targets,
        name=dataset_name,
        train=train,
        image_mode=f"POINTS_{num_features}d",
        transform=transform
    )
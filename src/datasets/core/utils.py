import torch
# from sklearn.preprocessing import StandardScaler
import os
import gzip
import pickle
from typing import Tuple, Optional, Callable, Any, Union
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pathlib import Path
import platform


def get_local_data_root(dataset_name: str):
    file_path = f"{os.getcwd()}{os.sep}data{os.sep}{dataset_name}"
    ensure_dir(file_path)
    return file_path


def get_package_data_root(dataset_name: str):
    return f"{os.path.dirname(os.path.dirname(__file__))}{os.sep}data{os.sep}{dataset_name}" # \todo get in a cleaner way to the data folder


def open_txt_gz(file_path: str, dtype: torch.dtype = torch.float) -> torch.tensor:
    with gzip.open(file_path, "rt") as f:
        data = np.loadtxt(f)
    return torch.as_tensor(data, dtype=dtype)


def ensure_dir(dirname: str):
    """Check whether a given directory was created; if not, create a new one.

    Args:
        dirname: string, path to the directory.
    """
    path = Path(dirname)
    if not path.is_dir():
        if platform.system() == 'Windows':
            os.makedirs(u"\\\\?\\" + os.path.join(os.getcwd(), dirname), exist_ok=False)
        else:
            path.mkdir(parents=True, exist_ok=False)


def generate_train_test_set_indices(dataset_name: str, train: bool, len_dataset: int, len_original_dataset: int):
    """
    Generates the indices of the train or test set of a dataset

    Note: (potentially) overlapping train and test datasets !
    """
    indices_path = f"{get_local_data_root(dataset_name)}{os.sep}{'train' if train else 'test'}_indices_size={len_dataset}.txt.gz"
    if os.path.exists(indices_path):
        indices = open_txt_gz(indices_path, dtype=torch.int32)
    else:
        indices = torch.randperm(len_original_dataset)[:len_dataset]
        save_txt_gz(indices_path, indices)
    return indices


def save_txt_gz(file_path: str, data: torch.Tensor):
    ensure_dir(os.path.dirname(file_path))
    with gzip.open(file_path, "wt") as f:
        np.savetxt(f, data.numpy())


def points_to_paths(x: torch.Tensor, path_length: int, path_step: float = None, y: torch.Tensor = None,
                    sample_path: bool = False, wiener_window: bool = False, fixed_window: bool = True, 
                    num_input_dims: int = 1, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    wiener process: w_k = w_k-1 + dw, dw  sim N(0, 1)
    :param x:
    :param y:
    :param path_length:
    :param path_step:
    :param sample_path:
    :param wiener_window:
    :param fixed_window:
    :param num_input_dims
    :param kwargs:
    :return:
    """
    input_size = x.shape[-num_input_dims:]
    batch_size = x.shape[:-num_input_dims]

    if y is None:
        y_paths = None
    else:
        output_size = y.shape[batch_size.__len__():]

    if sample_path:
        in_features_flat = input_size.numel()
        x = x.flatten(-num_input_dims)

        p = torch.ones(batch_size[-1]) * (1/batch_size[-1])
        idxs = p.multinomial(num_samples=batch_size[-1] * path_length, replacement=True).unsqueeze(-1)
        if batch_size.__len__() > 1:
            idxs.unsqueeze(0)

        x_paths = torch.gather(x, -2, idxs.expand(batch_size[:-1] + (-1, in_features_flat))).reshape(
            batch_size + (path_length,) + input_size)
        if y is not None:
            y_paths = torch.gather(y, -(output_size.__len__() + 1),
                                   idxs.expand(batch_size[:-1] + (-1, ) + output_size)).reshape(
                batch_size + (path_length, ) + output_size)
    elif num_input_dims > 1:
        raise NotImplementedError("wiener_window and fixed window not implemented for num_input_dim>1")
    elif wiener_window:
        triangle = torch.ones((path_length, path_length)).tril()
        dw = torch.randn(batch_size + (path_length,) + input_size)
        dw[..., 0, :] = 0.
        w = torch.einsum('lp,...pf->...lf', triangle, dw)
        x_paths = x.unsqueeze(-2).expand(batch_size + (path_length,) + input_size) + w
        if y is not None:
            y_paths = torch.zeros(batch_size + (path_length, ) + output_size).fill_(torch.nan)
    elif fixed_window:
        window = torch.arange(-path_length // 2 + 1, path_length // 2 + 1) * path_step
        x_paths = x.unsqueeze(-2).expand(batch_size + (path_length, ) + input_size) + window.unsqueeze(-1)
        if y is not None:
            y_paths = torch.zeros(batch_size + (path_length, ) + output_size).fill_(torch.nan)
    else:
        raise NotImplementedError

    return x_paths, y_paths


def pickle_dump(obj, tag):
    pickle_out = open("{}.pickle".format(tag), "wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()


def pickle_load(tag):
    if not ".npy" in tag or ".pickle" in tag:
        tag = f"{tag}.pickle"
    pickle_in = open(tag, "rb")
    if "npy" in tag:
        to_return = np.load(pickle_in)
    else:
        to_return = pickle.load(pickle_in)
    pickle_in.close()
    return to_return


def noise(size: tuple, sigma):
    return torch.randn(size) * sigma


def rand(size, l, u):
    return (u - l) * torch.rand(size) + l


def xdata(dataset_size: int, l: float, u: float, generate_paths: bool, generate_random: bool = True, **kwargs):
    if generate_random:
        x = (l - u) * torch.rand(dataset_size, 1) + u
    else:
        x = torch.linspace(l, u, int(dataset_size)).view(-1, 1)

    if generate_paths:
        return points_to_paths(x=x, **kwargs)
    else:
        return x


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    # return torch.eye(num_classes, dtype=torch.uint8)[y]
    return torch.eye(num_classes)[y]


def dataset_points_to_paths(x: torch.Tensor, y: torch.Tensor, path_length: int, sample_path: bool, **kwargs) -> \
        (torch.tensor, torch.tensor): # \todo to be removed
    """
    :param x: Size(batch_size, in_features)
    :param y: Size(batch_size, out_features)
    :param path_length
    :param sample_path
    """
    x_paths = points_to_paths(x, path_length=path_length, sample_path=sample_path, **kwargs)
    y_paths = torch.empty(x_paths.shape[:-1] + (y.shape[-1],)).fill_(torch.nan)
    return x_paths, y_paths


# def scale_data(scalar, data: np.array, scale: bool = True) -> torch.tensor:
#     if scale:
#         return torch.from_numpy(scalar.transform(data)).type(torch.float32)
#     else:
#         return torch.from_numpy(data).type(torch.float32)
#
#
# def post_process_data(pre_processer_x: StandardScaler, pre_processer_y: StandardScaler, x: np.array, y: np.array,
#                       scale: bool = True, generate_paths: bool = False, **kwargs):
#     x = scale_data(pre_processer_x, x, scale=scale)
#     y = scale_data(pre_processer_y, y, scale=scale)
#     if generate_paths:
#         x, y = dataset_points_to_paths(x, y, **kwargs)
#     return x, y


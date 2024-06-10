import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle
from typing import Tuple, Optional, Callable, Any, Union
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class FlattenTransform:
    def __init__(self, start_dim: int = 0, end_dim: int = -1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.flatten(start_dim=self.start_dim, end_dim=self.end_dim)


class MoveChannelTransform:
    def __init__(self, source: int = -1, destination: int = 0):
        self.source = source
        self.destination = destination

    def __call__(self, tensor: torch.Tensor):
        return tensor.movedim(self.source, self.destination)


class NormalizeTransform:
    """
    Apply z-score normalization on a given data.
    """
    def __init__(self, mean, std, eps: float = 1e-10):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, sample):
        return (sample - self.mean) / (self.std + self.eps)


def points_to_paths(x: torch.Tensor, path_length: int, path_step: float = None, y: torch.Tensor = None,
                    sample_path: bool = False, wiener_window: bool = False, fixed_window: bool = True, 
                    num_input_dims: int = 1, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    wiener process: w_k = w_k-1 + dw, dw \ sim N(0, 1)
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


def noise(size: tuple, sigma):
    return torch.randn(size) * sigma


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


def scale_data(scalar, data: np.array, scale: bool = True) -> torch.tensor:
    if scale:
        return torch.from_numpy(scalar.transform(data)).type(torch.float32)
    else:
        return torch.from_numpy(data).type(torch.float32)


def post_process_data(pre_processer_x: StandardScaler, pre_processer_y: StandardScaler, x: np.array, y: np.array,
                      scale: bool = True, generate_paths: bool = False, **kwargs):
    x = scale_data(pre_processer_x, x, scale=scale)
    y = scale_data(pre_processer_y, y, scale=scale)
    if generate_paths:
        x, y = dataset_points_to_paths(x, y, **kwargs)
    return x, y


class UCIRegressionDataset(Dataset):  # \todo use TensorDataset?
    def __init__(self, dataset_name: str, train: bool = True, paths: bool = False, ood: bool = False,
                 input_transform: Optional[Callable] = None, output_transform: Optional[Callable] = None,
                 **kwargs):
        self.train = train
        self.ood = ood
        self.paths = paths
        self.dataset_name = dataset_name
        self.input_transform, self.output_transform = input_transform, output_transform
        self.inputs, self.outputs = self._load_data(**kwargs)

    def _load_data(self, in_features: int, out_features: int, len_dataset: Optional[int] = None,
                   ratio_ood_samples: float = 0.5, uniform_ood_sampling: bool = False,
                   uniform_ood_sapling_dim: int = 0, random_seed: int = 0, **kwargs):
        folder_root = f"{os.path.dirname(__file__)}{os.sep}{self.dataset_name}"
        data_root = f"{folder_root}{os.sep}data.txt.gz"
        if os.path.exists(data_root):
            data = np.loadtxt(data_root)
            data = torch.from_numpy(data).to(torch.float32)
            inputs, outputs = data[:, :-out_features], data[:, -out_features:]
        else:
            raise NotImplementedError('data file not available')

        # ## temp
        # len_dataset_plot = 100
        # len_dataset_train = 7406
        # len_dataset_test = 128
        # for random_seed in range(10):
        #     indices_root_plot = f"{folder_root}{os.sep}{'train'}_indices_size={len_dataset_plot}_seed={random_seed}.txt.gz"
        #     indices_root_train = f"{folder_root}{os.sep}{'train'}_indices_size={len_dataset_train}_seed={random_seed}.txt.gz"
        #     indices_root_test = f"{folder_root}{os.sep}{'test'}_indices_size={len_dataset_test}_seed={random_seed}.txt.gz"
        #     indices = np.arange(inputs.shape[0])
        #     np.random.shuffle(indices)
        #
        #     indices_plot = indices[-len_dataset_plot:]
        #     indices_train = indices[:len_dataset_train]
        #     indices_test = indices[-len_dataset_test:]
        #
        #     np.savetxt(indices_root_plot, indices_plot)
        #     np.savetxt(indices_root_train, indices_train)
        #     np.savetxt(indices_root_test, indices_test)

        if len_dataset is None:
            len_dataset = inputs.shape[0]
        else:
            len_dataset = min(inputs.shape[0], len_dataset)

        if len_dataset == inputs.shape[0]:
            indices_root = f"{folder_root}{os.sep}{'train' if self.train else 'test'}_indices_size={len_dataset}.txt.gz"
            if os.path.exists(indices_root):
                indices = np.loadtxt(indices_root).astype(int)
            else:
                indices = np.arange(inputs.shape[0])
                np.random.shuffle(indices)
                indices = indices[:len_dataset]
                np.savetxt(indices_root, indices)

            inputs, outputs = inputs[indices], outputs[indices]

        if self.ood:
            if uniform_ood_sampling:
                # set all dimensions other than uniform_ood_sapling_dim to means
                domain_sampling_dim = [inputs[:, uniform_ood_sapling_dim].min(),
                                       inputs[:, uniform_ood_sapling_dim].max()]
                inputs = inputs.mean(0)
                inputs = inputs.unsqueeze(0).expand(len_dataset, in_features).clone()
                inputs[:, uniform_ood_sapling_dim] = torch.linspace(*domain_sampling_dim, len_dataset)
                outputs = torch.zeros(len_dataset, out_features).fill_(torch.nan)
            else:
                num_ood_samples = int(inputs.shape[0] * ratio_ood_samples / (1 - ratio_ood_samples))
                domain = [inputs.min(0).values, inputs.max(0).values]

                ood_inputs = domain[0] + (domain[1] - domain[0]) * torch.rand(num_ood_samples, in_features)
                ood_outputs = torch.zeros(num_ood_samples, out_features).fill_(torch.nan)

                inputs = torch.cat((inputs, ood_inputs))
                outputs = torch.cat((outputs, ood_outputs))

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


class ClassificationDataset(Dataset):  # \todo use TensorDataset?
    def __init__(self, train: bool, paths: bool, data: torch.Tensor, targets: torch.Tensor, len_dataset: int,
                 mode: Optional[str], num_classes: int, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, **kwargs):
        self.train = train
        self.paths = paths
        self.mode = mode
        self.img_size = data.shape[1:]
        self.data, self.targets = self._initialize_data(data, targets, len_dataset, num_classes, **kwargs)
        self.transform = transform
        self.target_transform = target_transform

    def _initialize_data(self, data: torch.Tensor, targets: torch.Tensor, len_dataset: int, num_classes, **kwargs):
        data, targets = data[:len_dataset], targets[:len_dataset]
        targets = to_categorical(targets, num_classes)

        if self.paths:
            return points_to_paths(x=data, y=targets, num_input_dims=data.dim() - 1, **kwargs)
        else:
            return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        img = self._scale_image(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _scale_image(self, img: torch.Tensor):  # \todo vectorize Image.fromarray
        if img.shape == self.img_size:
            img = Image.fromarray(img.numpy(), mode=self.mode)
            return transforms.functional.to_tensor(img).squeeze()
        else:
            return torch.stack([self._scale_image(elem) for elem in img])

    def __len__(self) -> int:
        return len(self.data)



# \todo depreciate:
def load_uci_dataset(dataset_name: str,  len_train_dataset: int, len_test_dataset: int, generate_ood: bool = False,
                     plot_dim: int = 0, in_features: int = 1, out_features: int = 1, **kwargs):
    folder_path = '{}/{}'.format(os.path.dirname(__file__), dataset_name)
    data_file_path = '{}/data.txt.gz'.format(folder_path)
    if os.path.exists(data_file_path):
        data = np.loadtxt(data_file_path)
    else:
        raise NotImplementedError('data file not available')

    x = data[:, :-out_features]; y = data[:, -out_features:]

    train_indices_path = '{}/train_indices_size={}.txt.gz'.format(folder_path, len_train_dataset)
    test_indices_path = '{}/test_indices_size={}.txt.gz'.format(folder_path, len_test_dataset)
    if not os.path.exists(train_indices_path) and not os.path.exists(test_indices_path):
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        indices_train = indices[:len_train_dataset]
        indices_test = indices[-len_test_dataset:]
        np.savetxt(train_indices_path, indices_train)
        np.savetxt(test_indices_path, indices_test)
    else:
        indices_train = np.loadtxt(train_indices_path).astype(int)
        indices_test = np.loadtxt(test_indices_path).astype(int)

    x_train, y_train = x[indices_train], y[indices_train]
    x_test, y_test = x[indices_test], y[indices_test]

    nr_plot_points = 10
    x_plot = np.zeros((nr_plot_points, in_features))
    x_plot[:, plot_dim] = np.linspace(x_train[:, plot_dim].min(), x_train[:, plot_dim].max(), nr_plot_points)
    y_plot = np.zeros((nr_plot_points, out_features))

    pre_processer_x = StandardScaler()
    pre_processer_x.fit(x_train)

    pre_processer_y = StandardScaler()
    pre_processer_y.fit(y_train)

    if generate_ood:
        # use x% datapoints using uniformly sampled point from the input domain # \todo specify desired set size
        ratio_ood_samples = 0.5
        nr_ood_train_samples = int(x_train.shape[0] * ratio_ood_samples / (1-ratio_ood_samples))
        nr_ood_test_samples = int(x_test.shape[0] * ratio_ood_samples / (1-ratio_ood_samples))
        l_domain, u_domain = x.min(0), x.max(0)

        x_train_ood = np.random.uniform(l_domain, u_domain, size=(nr_ood_train_samples, in_features))
        y_train_ood = np.full((nr_ood_train_samples, 1), np.nan)
        x_train = np.concatenate((x_train, x_train_ood))
        y_train = np.concatenate((y_train, y_train_ood))

        x_test_ood = np.random.uniform(l_domain, u_domain, size=(nr_ood_test_samples, in_features))
        y_test_ood = np.full((nr_ood_test_samples, 1), np.nan)
        x_test = np.concatenate((x_test, x_test_ood))
        y_test = np.concatenate((y_test, y_test_ood))

    scale = True
    x_train, y_train = post_process_data(pre_processer_x, pre_processer_y, x_train, y_train, scale=scale, **kwargs)
    x_test, y_test = post_process_data(pre_processer_x, pre_processer_y, x_test, y_test, scale=scale, **kwargs)
    x_plot, y_plot = post_process_data(pre_processer_x, pre_processer_y, x_plot, y_plot, scale=scale, **kwargs)

    input_size = x_train.shape[-1]
    output_size = y_train.shape[-1]
    return x_train, y_train, x_test, y_test, x_plot, y_plot, input_size, output_size




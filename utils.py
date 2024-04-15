import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle


def points_to_paths(x, path_length: int, path_window_step: float, sampled_window: bool = False,
                    wiener_window: bool = False, fixed_window: bool = True, **kwargs):
    """
    wiener process: w_k = w_k-1 + dw, dw \ sim N(0, 1)
    :param x:
    :param path_length:
    :param path_window_step:
    :param sampled_window:
    :param wiener_window:
    :param fixed_window:
    :param kwargs:
    :return:
    """
    batch_size = x.shape[:-1]
    features = x.shape[-1]
    if sampled_window:
        p = torch.ones(batch_size.numel()) * (1/batch_size.numel())
        idxs = p.multinomial(num_samples=batch_size.numel() * path_length, replacement=True)
        x_paths = x.reshape(batch_size.numel(), features)[idxs].reshape(batch_size + (path_length, features))
    elif wiener_window:
        triangle = torch.ones((path_length, path_length)).tril()
        dw = torch.randn(batch_size + (path_length,) + (features,))
        dw[..., 0, :] = 0.
        w = torch.einsum('lp,...pf->...lf', triangle, dw)
        x_paths = x[..., None, :].repeat(len(batch_size) * (1,) + (path_length, ) + (1,)) + w
    elif fixed_window:
        window = torch.arange(-path_length // 2 + 1, path_length // 2 + 1) * path_window_step
        x_paths = torch.hstack(path_length * (x[..., None, :],)) + torch.hstack(features * (window[..., None], ))
    else:
        raise NotImplementedError
    return x_paths


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


def dataset_points_to_paths(x: torch.Tensor, y: torch.Tensor, path_length: int, sampled_window: bool, **kwargs) -> \
        (torch.tensor, torch.tensor):
    """
    :param x: Size(batch_size, in_features)
    :param y: Size(batch_size, out_features)
    :param path_length
    :param sampled_window
    """
    x_paths = points_to_paths(x, path_length=path_length, sampled_window=sampled_window, **kwargs)
    y_paths = torch.empty(x_paths.shape[:-1] + (y.shape[-1],)).fill_(torch.nan)
    return x_paths, y_paths


def scale_data(scalar, data: np.array, scale: bool = True) -> torch.tensor:
    if scale:
        return torch.from_numpy(scalar.transform(data)).type(torch.float32)
    else:
        return torch.from_numpy(data).type(torch.float32)


def post_process_data(x_scalar: StandardScaler, y_scalar: StandardScaler, x: torch.Tensor, y: torch.Tensor,
                      scale: bool = True, generate_paths: bool = False, **kwargs):
    x = scale_data(x_scalar, x, scale=scale)
    y = scale_data(y_scalar, y, scale=scale)
    if generate_paths:
        x, y = dataset_points_to_paths(x, y, **kwargs)
    return x, y


def load_uci_dataset(dataset_name: str,  dataset_size_train: int, dataset_size_test: int, generate_ood: bool = False,
                     plot_dim: int = 0, in_features: int = 1, out_features: int = 1, **kwargs):
    folder_path = '{}/{}'.format(os.path.dirname(__file__), dataset_name)
    data_file_path = '{}/data.txt.gz'.format(folder_path)
    if os.path.exists(data_file_path):
        data = np.loadtxt(data_file_path)
    else:
        raise NotImplementedError('data file not available')

    x = data[:, :-out_features]; y = data[:, -out_features:]

    train_indices_path = '{}/train_indices_size={}.txt.gz'.format(folder_path, dataset_size_train)
    test_indices_path = '{}/test_indices_size={}.txt.gz'.format(folder_path, dataset_size_test)
    if not os.path.exists(train_indices_path) and not os.path.exists(test_indices_path):
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        indices_train = indices[:dataset_size_train]
        indices_test = indices[-dataset_size_test:]
        np.savetxt(train_indices_path, indices_train)
        np.savetxt(test_indices_path, indices_test)
    else:
        indices_train = np.loadtxt(train_indices_path).astype(int)
        indices_test = np.loadtxt(test_indices_path).astype(int)

    x_train, y_train = x[indices_train], y[indices_train]
    x_test, y_test = x[indices_test], y[indices_test]

    # x_train = x[:dataset_size_train, :]
    # y_train = y[:dataset_size_train]
    #
    # x_test = x[-dataset_size_test:, :]
    # y_test = y[-dataset_size_test:]

    nr_plot_points = 10
    x_plot = np.zeros((nr_plot_points, in_features))
    x_plot[:, plot_dim] = np.linspace(x_train[:, plot_dim].min(), x_train[:, plot_dim].max(), nr_plot_points)
    y_plot = np.zeros((nr_plot_points, out_features))

    x_scalar = StandardScaler()
    x_scalar.fit(x_train)

    y_scalar = StandardScaler()
    y_scalar.fit(y_train)

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
    x_train, y_train = post_process_data(x_scalar, y_scalar, x_train, y_train, scale=scale, **kwargs)
    x_test, y_test = post_process_data(x_scalar, y_scalar, x_test, y_test, scale=scale, **kwargs)
    x_plot, y_plot = post_process_data(x_scalar, y_scalar, x_plot, y_plot, scale=scale, **kwargs)

    input_shape = x_train.shape[-1]
    output_shape = y_train.shape[-1]
    return x_train, y_train, x_test, y_test, x_plot, y_plot, input_shape, output_shape

import numpy as np
import os
from sklearn.preprocessing import StandardScaler

from .utils import post_process_data


def load_luca(dataset_size_train: int = None, dataset_size_test: int = None, **kwargs):
    folder_path = '{}/luca'.format(os.path.dirname(__file__))
    data_file_path = '{}/data.txt.gz'.format(folder_path)
    if os.path.exists(data_file_path):
        data = np.loadtxt(data_file_path)
    else:
        raise NotImplementedError('data file not available')

    x = data[:, :-1]; y = data[:, -1:]

    if dataset_size_train is None:
        dataset_size_train = x.shape[0]
    if dataset_size_test is None:
        dataset_size_test = x.shape[0]

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

    pre_processer_x = StandardScaler()
    pre_processer_x.fit(x_train)

    pre_processer_y = StandardScaler()
    pre_processer_y.fit(y_train)

    scale = False
    x_train, y_train = post_process_data(pre_processer_x, pre_processer_y, x_train, y_train, scale=scale, **kwargs)
    x_test, y_test = post_process_data(pre_processer_x, pre_processer_y, x_test, y_test, scale=scale, **kwargs)

    sorting_mask = x_test.sort(dim=0).indices[:, 0]  # order test data w.r.t. 1st dimension
    x_plot, y_plot = x_test[sorting_mask], y_test[sorting_mask]

    input_shape = x_train.shape[-1]
    output_shape = y_train.shape[-1]
    return x_train, y_train, x_test, y_test, x_plot, y_plot, input_shape, output_shape

import torch
import math


def f(x: torch.Tensor):
    return x + 0.3 * torch.sin(2 * math.pi * x) + 0.3 * torch.sin(4 * math.pi * x)


def load_uva_tutorial(dataset_size_train: int, dataset_size_test: int, generate_ood: bool = False, **kwargs):
    x_obs = torch.cat([torch.linspace(-0.2, 0.2, dataset_size_train//2),
                       torch.linspace(0.6, 1, dataset_size_train//2)]).unsqueeze(-1)
    noise = 0.02 * torch.randn(x_obs.shape)  # sigma 0.2
    y_obs = f(x_obs + noise)

    x_true = torch.linspace(-0.5, 1.5, 1000).unsqueeze(-1)
    y_true = f(x_true)

    x_train, y_train = x_obs, y_obs
    x_test, y_test = x_obs, y_obs
    x_plot, y_plot = x_true, y_true

    input_shape = x_train.shape[-1]
    output_shape = y_train.shape[-1]

    return x_train, y_train, x_test, y_test, x_plot, y_plot, input_shape, output_shape

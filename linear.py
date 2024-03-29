from .utils import noise, xdata


def linear(x, sigma: float = 0.1):
    return 0.5 * x + noise(x.shape, sigma=sigma)


def load_linear(train_dataset_size: int = 2**10, test_dataset_size: int = 2**10, **kwargs):
    x_train = xdata(dataset_size=train_dataset_size, generate_random=True, **kwargs)
    y_train = linear(x_train)
    x_test = xdata(dataset_size=test_dataset_size, generate_random=True, **kwargs)
    y_test = linear(x_test)

    input_shape, output_shape = x_train.shape[-1], y_train.shape[-1]
    return x_train, y_train, x_test, y_test, input_shape, output_shape
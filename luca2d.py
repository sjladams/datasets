from .luca import load_luca


def load_luca2d(**kwargs):
    x_train, y_train, x_test, y_test, x_plot, y_plot, input_size, output_size = load_luca(**kwargs)

    y_train = y_train[..., None].expand(y_train.shape + (2,)).squeeze()
    y_test = y_test[..., None].expand(y_test.shape + (2,)).squeeze()
    y_plot = y_plot[..., None].expand(y_plot.shape + (2,)).squeeze()

    input_size = x_train.shape[-1]
    output_size = y_train.shape[-1]
    return x_train, y_train, x_test, y_test, x_plot, y_plot, input_size, output_size

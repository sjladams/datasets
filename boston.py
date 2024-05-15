from .utils import load_uci_dataset


def load_boston(dataset_size_train=442, dataset_size_test=64, **kwargs):
    return load_uci_dataset(dataset_name='boston', in_features=13, out_features=1,
                            dataset_size_train=dataset_size_train, dataset_size_test=dataset_size_test, **kwargs)

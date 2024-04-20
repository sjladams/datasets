from .utils import load_uci_dataset


def load_energy(**kwargs):
    return load_uci_dataset(dataset_name='energy', in_features=8, out_features=1, dataset_size_train=640,
                            dataset_size_test=128, **kwargs)
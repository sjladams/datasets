from .utils import load_uci_dataset


def load_boston(**kwargs):
    return load_uci_dataset(dataset_name='boston', in_features=13, out_features=1, **kwargs)

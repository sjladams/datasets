from .utils import load_uci_dataset


def load_concrete(**kwargs):
    return load_uci_dataset(dataset_name='concrete', in_features=8, out_features=1, **kwargs)

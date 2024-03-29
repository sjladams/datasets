from .utils import load_uci_dataset


def load_naval(**kwargs):
    return load_uci_dataset(dataset_name='naval', in_features=16, out_features=2, **kwargs)

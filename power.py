from .utils import load_uci_dataset


def load_power(**kwargs):
    return load_uci_dataset(dataset_name='power', in_features=4, out_features=1, **kwargs)
from .utils import load_uci_dataset


def load_kin8nm(**kwargs):
    return load_uci_dataset(dataset_name='kin8nm', in_features=8, out_features=1, **kwargs)

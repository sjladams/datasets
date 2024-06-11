


def load_wine(**kwargs):
    return load_uci_dataset(dataset_name='wine', in_features=11, out_features=1, **kwargs)
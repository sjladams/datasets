


def load_protein(**kwargs):
    return load_uci_dataset(dataset_name='protein', in_features=9, out_features=1, **kwargs)
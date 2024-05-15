from .utils import load_uci_dataset


def load_kin8nm(dataset_size_train=7406, dataset_size_test=786, **kwargs):
    return load_uci_dataset(dataset_name='kin8nm', in_features=8, out_features=1,
                            dataset_size_train=dataset_size_train, dataset_size_test=dataset_size_test, **kwargs)

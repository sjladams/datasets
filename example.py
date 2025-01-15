from datasets import get_dataset

# from torch.utils.data import DataLoader, Dataset
# def get_data_loader(dataset_name: str, batch_size: int, shuffle: bool = False, paths=False, **kwargs) -> \
#         Tuple[DataLoader, torch.Size, torch.Size]:
#     ds, input_size, output_size = get_dataset(dataset_name=dataset_name, paths=paths, **kwargs)
#     loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle,
#                         worker_init_fn=np.random.seed(0), num_workers=0)
#     return loader, input_size, output_size


if __name__ == '__main__':
    ds = get_dataset(
        dataset_name='snelson',
        train=True,
        len_dataset=50,
        in_features=1,
        out_features=1,
        # flatten=True
    )

    print('test')
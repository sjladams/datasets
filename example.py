from datasets import get_dataset

if __name__ == '__main__':
    ds = get_dataset(
        dataset_name="ellipsoid_split",
        train=True,
        # len_dataset=50,
        in_features=2,
        # out_features=1,
        # flatten=True
    )

print('test')

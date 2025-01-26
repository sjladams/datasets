import torch
from typing import Tuple, Optional, Any, Union


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data: torch.Tensor,
                 targets: torch.Tensor,
                 name: str,
                 train: bool = True,
                 ood: bool = False,
                 transform: Optional[torch.nn.Module] = None,
                 target_transform: Optional[torch.nn.Module] = None):
        self.data, self.targets = data, targets
        self.name = name
        self.train = train
        self.ood = ood
        self.transform, self.target_transform = transform, target_transform

    def __getitem__(self, index: Union[int, list, torch.Tensor]) -> Tuple[Any, Any]:
        """
        batched indexing and transformations
        """
        data, target = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def data_size(self):
        return self.__getitem__(0)[0].size()

    @property
    def target_size(self):
        return None

    def subset(self, indices: torch.Tensor):
        return Dataset(
            data=self.data[indices],
            targets=self.targets[indices],
            name=self.name,
            train=self.train,
            ood=self.ood,
            transform=self.transform,
            target_transform=self.target_transform
        )


class ClassificationDataset(Dataset):
    def __init__(self, image_mode: str, **kwargs):
        self.image_mode = image_mode
        super().__init__(**kwargs)

        self._num_classes = len(torch.unique(self.targets))
        assert self._num_classes == self.targets.max().item() + 1, 'Class indices must be contiguous'

    @property
    def target_size(self):
        return torch.Size((self._num_classes,))


class RegressionDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def target_size(self):
        return self.__getitem__(0)[1].size()


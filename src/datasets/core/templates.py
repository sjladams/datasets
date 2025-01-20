import torch
from typing import Tuple, Optional, Any, Union


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data: torch.Tensor,
                 targets: torch.Tensor,
                 train: bool = True,
                 transform: Optional[torch.nn.Module] = None,
                 target_transform: Optional[torch.nn.Module] = None):
        self.data, self.targets = data, targets
        self.train = train
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
    def input_size(self):
        return self.__getitem__(0)[0].shape

    @property
    def output_size(self):
        return self.__getitem__(0)[1].shape


class ClassificationDataset(Dataset):
    def __init__(self, image_mode: str, **kwargs):
        self.image_mode = image_mode
        super().__init__(**kwargs)


class RegressionDataset(Dataset):
    def __init__(self, ood: bool = False, **kwargs):
        self.ood = ood
        super().__init__(**kwargs)


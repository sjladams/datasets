from typing import Optional, Union
import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor


class Flatten(torch.nn.modules.Flatten):
    def __init__(self, image_mode: str):
        """
        Custom Flatten layer that adjusts start_dim based on the PIL image mode.

        Args:
            image_mode (str): The mode of the PIL image (e.g., 'L', 'RGB').
        """
        # Map modes to their respective start_dim (Because of usage of EnsureChannel, this map is redundant!)
        mode_to_start_dim = {
            "L": -3,         # Grayscale: (1, H, W) -> Flatten H and W
            # "1": -2,         # Binary: (1, H, W) -> Flatten H and W
            "RGB": -3,       # True color: (C, H, W) -> Flatten C, H, W
            # "RGBA": -3,      # True color with alpha: (C, H, W) -> Flatten C, H, W
            # "CMYK": -3,      # Printing colors: (C, H, W) -> Flatten C, H, W
            # "P": -2,         # Palette-based: (H, W) -> Flatten H and W
            # "I": -2,         # 32-bit integer: (H, W) -> Flatten H and W
            # "F": -2,         # Floating-point: (H, W) -> Flatten H and W
            # "YCbCr": -3,     # Luminance/chrominance: (C, H, W) -> Flatten C, H, W
            # "LAB": -3,       # LAB color space: (C, H, W) -> Flatten C, H, W
            # "HSV": -3,       # Hue/Saturation/Value: (C, H, W) -> Flatten C, H, W
            # "RGBX": -3       # True color with padding: (C, H, W) -> Flatten C, H, W
        }

        if image_mode not in mode_to_start_dim:
            raise ValueError(f"Unsupported image mode: {image_mode}")

        start_dim = mode_to_start_dim[image_mode]
        super().__init__(start_dim=start_dim)


class NormalizeImage(transforms.Normalize):
    """
    Apply z-score normalization on a given image data.
    """
    def __init__(self, mean, std):
        super().__init__(mean, std)


class NormalizeNumerical(torch.nn.modules.Module):
    """
    Apply z-score normalization on a given image data.
    """
    def __init__(self, mean: Union[float, torch.Tensor] = 0., std: Union[float, torch.Tensor] = 1.):
        super(NormalizeNumerical, self).__init__()
        self.mean = torch.as_tensor(mean).type(torch.float32)
        self.std = torch.as_tensor(std).type(torch.float32)

        # In case the data takes a fixed value in some dimension, set std to 1 to avoid division by zero
        self.std[self.std == 0.] = 1.

    def forward(self, x: torch.Tensor):
        return (x - self.mean) / self.std


class Float(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data: torch.Tensor):
        return data.float()


class EnsureChannel(torch.nn.Module):
    def __init__(self, size: Union[tuple, torch.Size]):
        if len(size) == 2:
            self.init_channel = True
        elif len(size) == 3:
            self.init_channel = False
        else:
            raise ValueError

        super().__init__()

    def forward(self, data: torch.Tensor):
        if self.init_channel:
            return data.unsqueeze(-3)
        else:
            return data


class MoveChannel:
    def __init__(self, source: int = -1, destination: int = 0):
        self.source = source
        self.destination = destination

    def __call__(self, tensor: torch.Tensor):
        return tensor.movedim(self.source, self.destination)

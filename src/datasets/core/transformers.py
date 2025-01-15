import torch
from torchvision.transforms import Normalize

class FlattenTransform(torch.nn.modules.Flatten):
    def __init__(self, image_mode: str):
        """
        Custom Flatten layer that adjusts start_dim based on the PIL image mode.

        Args:
            image_mode (str): The mode of the PIL image (e.g., 'L', 'RGB').
        """
        # Map modes to their respective start_dim
        mode_to_start_dim = {
            "L": -2,         # Grayscale: (C, H, W) -> Flatten H and W
            "1": -2,         # Binary: (C, H, W) -> Flatten H and W
            "RGB": -3,       # True color: (C, H, W) -> Flatten C, H, W
            "RGBA": -3,      # True color with alpha: (C, H, W) -> Flatten C, H, W
            "CMYK": -3,      # Printing colors: (C, H, W) -> Flatten C, H, W
            "P": -2,         # Palette-based: (C, H, W) -> Flatten H and W
            "I": -2,         # 32-bit integer: (C, H, W) -> Flatten H and W
            "F": -2,         # Floating-point: (C, H, W) -> Flatten H and W
            "YCbCr": -3,     # Luminance/chrominance: (C, H, W) -> Flatten C, H, W
            "LAB": -3,       # LAB color space: (C, H, W) -> Flatten C, H, W
            "HSV": -3,       # Hue/Saturation/Value: (C, H, W) -> Flatten C, H, W
            "RGBX": -3       # True color with padding: (C, H, W) -> Flatten C, H, W
        }

        if image_mode not in mode_to_start_dim:
            raise ValueError(f"Unsupported image mode: {image_mode}")

        start_dim = mode_to_start_dim[image_mode]
        super().__init__(start_dim=start_dim)


class NormalizeTransform(Normalize):
    """
    Apply z-score normalization on a given data.
    """
    def __init__(self, mean, std):
        super().__init__(mean, std)


class MoveChannelTransform:
    def __init__(self, source: int = -1, destination: int = 0):
        self.source = source
        self.destination = destination

    def __call__(self, tensor: torch.Tensor):
        return tensor.movedim(self.source, self.destination)

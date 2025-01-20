from .torchvision import load_torchvision
from .uci import load_uci
from .custom_regression import load_custom_regression
from .other_regression import load_other_regression
from .med_mnist import load_medmnist

__all__ = [
    'load_torchvision',
    'load_uci',
    'load_custom_regression',
    'load_other_regression',
    'load_medmnist'
]
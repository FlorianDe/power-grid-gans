import sys
import torch
import torchvision
import numpy as np

dependencies = {
    'torch': torch,
    'torchvision': torchvision,
    'numpy': np
}
if __name__ == "__main__":
    print("Dependencies")
    print(f'Using Python: {sys.prefix}')
    for name, dependency in dependencies.items():
        print(f'{name}: {dependency.__version__}')
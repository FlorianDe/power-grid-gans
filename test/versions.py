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
    pythonVersion = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}-{sys.version_info.releaselevel}'
    print("Dependencies")
    print(f'Using Python: Version {pythonVersion} from {sys.prefix}')
    for name, dependency in dependencies.items():
        print(f'{name}: {dependency.__version__}')
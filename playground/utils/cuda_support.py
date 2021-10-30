import torch

from test.utils import LOG

if __name__ == "__main__":
    LOG.info(f'Cuda support available: {torch.cuda.is_available():}')
    # Try converting/creating a tensor on the GPU, throws an error if not possible!
    # Possible Errors:
    # 1. "Torch not compiled with CUDA enabled" <-- Check README.md
    try:
        torch.zeros(1).cuda()
    except AssertionError as err:
        LOG.info(f'Reason: {err}')



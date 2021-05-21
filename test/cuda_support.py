import torch

if __name__ == "__main__":
    print(f'Cuda support available: {torch.cuda.is_available():}')
    # Try converting/creating a tensor on the GPU, throws an error if not possible!
    # Possible Errors:
    # 1. "Torch not compiled with CUDA enabled" <-- Check README.md
    torch.zeros(1).cuda()

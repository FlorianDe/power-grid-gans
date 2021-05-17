import torch

if __name__ == "__main__":
    print(f'Cuda support available: {torch.cuda.is_available():}')
    torch.zeros(1).cuda()

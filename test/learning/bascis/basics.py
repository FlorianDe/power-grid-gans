import torch
import torch.nn as nn
import numpy as np

# Creation of tensors
m = np.random.rand(3, 2)
t = torch.tensor(m, dtype=torch.float16)
print("Tensor: ", t)

# Scalar tensors
sumTensor = t.sum()
print("Sum: ", sumTensor.item())

# GPU vs CPU
a = torch.FloatTensor([2, 3])
print("FloatTensor: ", a)
# ca = a.cuda() # Not working on mac because no cuda GPU ofc
# print("FloatTensorCuda: ", ca)

# Gradients
v1 = torch.tensor([1.0, 1.0], requires_grad=True)
v2 = torch.tensor([2.0, 2.0])
v_sum = v1 + v2
v_res = (v_sum*2).sum()
for key, value in {'v1': v1, 'v2': v2, 'v_sum':v_sum, 'v_res': v_res}.items():
    print(f'{key}: isLeaf: {value.is_leaf}, reqGrad: {value.requires_grad}')
print(f'v1.grad: {v1.grad}')
v_res.backward()
print(f'v1.grad: {v1.grad}')
print(f'v2.grad: {v2.grad}')

# Basic NN
feedForwardLayer = nn.Linear(2, 5)  # 2 in, 5 out
fflInp = torch.FloatTensor([1, 2])
print(f'ffl Output: {feedForwardLayer(fflInp)}')

# 3-Layer-NN
nn3Lay = nn.Sequential(
    feedForwardLayer,
    nn.ReLU(),
    nn.Linear(5, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.Dropout(p=0.3),
    nn.Softmax(dim=1)
)
print(f'nn3Lay: {nn3Lay}')
print(f'nn3LayOutput: {nn3Lay(torch.FloatTensor([[1, 2]]))}')

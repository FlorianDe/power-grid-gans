import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.fnn import LinearNet
from models.cnn import ConvolutionalNet

if __name__ == "__main__":
    mnist_root_dir = os.path.join(sys.prefix, 'mnist_data')
    epochs = 10
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    mnist_trainset = datasets.MNIST(root=mnist_root_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(mnist_trainset, batch_size=batch_size_train, shuffle=True)

    mnist_testset = datasets.MNIST(root=mnist_root_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(mnist_testset, batch_size=batch_size_test, shuffle=True)

    assert len(mnist_trainset.data) > 0, "The training set has no elements!"
    sample_data = mnist_trainset.data[0]
    assert len(sample_data.shape) == 2, "The training set data has the wrong shape, it should be [x, y]"
    img_size = tuple(sample_data.shape)
    # net = LinearNet(img_size[0]*img_size[1], [347, 49])
    net = ConvolutionalNet(img_size, None, [50])
    cross_el = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        net.train()
        print(f'epoch: {epoch}')
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = net(data)
            loss = cross_el(output, target)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            correct, total = 0, 0
            for data in test_loader:
                x, y = data
                output = net(x)
                for idx, i in enumerate(output):
                    if torch.argmax(i) == y[idx]:
                        correct += 1
                    total += 1
            print(f'accuracy: {round(correct / total, 3)}')

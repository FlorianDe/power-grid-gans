import sys
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from playground.pytorch.classification.digits.models.digit_fnn import DigitLinearNet

from src.net.dynamic import ConvolutionalNet, RecurrentNet
from src.net import CustomModule


@dataclass
class NetHolder:
    net: CustomModule
    optimizer: optim.Optimizer

    def __iter__(self):
        return iter((self.net, self.optimizer))


if __name__ == "__main__":
    mnist_root_dir = os.path.join(sys.prefix, 'mnist_data')
    epochs = 10
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.001

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # dataset shape [batch, 1, WIDTH, HEIGHT]
    mnist_trainset = datasets.MNIST(root=mnist_root_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(mnist_trainset, batch_size=batch_size_train, shuffle=True)

    mnist_testset = datasets.MNIST(root=mnist_root_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(mnist_testset, batch_size=batch_size_test, shuffle=True)

    assert len(mnist_trainset.data) > 0, "The training set has no elements!"
    sample_data = mnist_trainset.data[0]
    assert len(sample_data.shape) == 2, "The training set data has the wrong shape, it should be [x, y]"
    img_size = tuple(sample_data.shape)
    fc_net = DigitLinearNet(img_size[0]*img_size[1], 10, [347, 49])
    cnn_net = ConvolutionalNet(img_size, None, [50])
    rnn_net = RecurrentNet(img_size[0], img_size[1], 2, 128)
    nets = {
        'fnn': NetHolder(fc_net, optim.Adam(fc_net.parameters(), lr=learning_rate)),
        'cnn': NetHolder(cnn_net, optim.Adam(cnn_net.parameters(), lr=learning_rate)),
        'rnn': NetHolder(rnn_net, optim.Adam(rnn_net.parameters(), lr=learning_rate))
    }

    cross_el = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for name, holder in nets.items():
            net = holder.net
            optimizer = holder.optimizer

            net.train()
            print(f'{name} epoch: {epoch}')
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = net(net.reshape(data))
                loss = cross_el(output, target)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                correct, total = 0, 0
                for data in test_loader:
                    x, y = data
                    output = net(net.reshape(x))
                    for idx, i in enumerate(output):
                        if torch.argmax(i) == y[idx]:
                            correct += 1
                        total += 1
                print(f'{name} accuracy: {round(correct / total, 3)}')

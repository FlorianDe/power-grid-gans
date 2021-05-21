import argparse
import math
import os
from datetime import datetime
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src.tensorboard.utils import TensorboardUtils, GraphPlotItem


# torch.autograd.set_detect_anomaly(True)


class PolynomialLayer(nn.Module):
    def __init__(self, _order: int, _device):
        super().__init__()
        self.device = _device
        if _order < 1:
            raise ValueError("Order has to be at least 1 for a linear regression!")
        self.params = nn.ParameterList([nn.Parameter(torch.randn((), device=self.device), requires_grad=True) for _ in range(_order + 1)])

    def forward(self, x):
        value = torch.zeros(x.size(), requires_grad=False, device=self.device)
        for order, coefficient in enumerate(self.params):
            value += (x.pow(order)).multiply(coefficient)
        return value

    def string(self):
        expressions = [""] * len(self.params)
        for order in range(len(self.params)):
            expressions[order] = f'{self.params[order].item()}*x^{order}'
        return 'y=' + ' + '.join(expressions)


class RegressionNet:
    def __init__(self, _net_name, _device, _order: int, _x_samples, _y_samples, _runs=1, _batch_size=50):
        self.model = PolynomialLayer(_order, _device)
        self.device = _device
        self.net_name = _net_name
        self.x_samples = _x_samples
        self.y_samples = _y_samples
        self.runs = _runs
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-6, momentum=0.97)
        self.train_loader = DataLoader(
            TensorDataset(_x_samples, _y_samples),
            batch_size=min(len(_x_samples), _batch_size),
            shuffle=True,
            # num_workers=2
        )

    def load(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def save(self, path):
        torch.save(self.model.state_dict(), f'{path}/model.pt')

    def calc(self, values):
        self.model.eval()  # Set the model to "eval" mode, alias for model.train(mode=False)
        with torch.no_grad():
            return self.model(values)

    def train(self, writer: SummaryWriter):
        self.model.train()  # Set the model to "train" mode
        for epoch in tqdm(range(self.runs)):
            running_loss = 0
            for i, data in enumerate(self.train_loader, 0):
                x_samples_batch, y_samples_batch = data
                y_pred = self.model(x_samples_batch)
                loss = self.criterion(y_pred, y_samples_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss

            if (self.runs / 100) < 2 or (epoch % math.ceil(self.runs / 100)) == 0:
                writer.add_scalar(
                    tag='training loss',
                    scalar_value=running_loss / len(self.train_loader),
                    global_step=epoch * len(self.train_loader)
                )
                TensorboardUtils.plot_graph_as_figure(
                    tag="function/comparison",
                    writer=writer,
                    plot_data=[
                        GraphPlotItem(
                            label="real",
                            x=self.x_samples.detach().numpy(),
                            y=self.y_samples.detach().numpy(),
                            color='c'
                        ),
                        GraphPlotItem(
                            label="pred",
                            x=self.x_samples.detach().numpy(),
                            y=self.calc(self.x_samples).detach().numpy(),
                            color='r'
                        ),
                    ],
                    global_step=epoch * len(self.train_loader)
                )

        polynomial_str = self.model.string()
        print(f'Result: {polynomial_str}')
        writer.add_text(net_name, polynomial_str, runs)
        self.save(self.net_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action='store_true', help="Enable cuda computation")
    parser.add_argument("--load", help="The model path which should be loaded.")
    parser.add_argument("--eval", help="Set the model to training mode else computation mode.")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(
        "Cuda support available, transferring all tensors to the gpu!" if torch.cuda.is_available() else
        "No cuda support available!"
    )
    print(f'Using default device for tensors: {device}')

    order = 6  # Should be saved too
    start = - math.pi
    end = math.pi
    func = torch.sin
    stepSize = 0.2
    steps = math.floor((end - start) / stepSize)
    batchSize = 10  # steps
    x_samples = torch.linspace(start, end, steps, device=device)
    y_samples = torch.sin(x_samples).to(device)
    runs = 100_000

    net_name = f'runs/regression_order{order:03}_batches{batchSize:04}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    writer = SummaryWriter(f'{net_name}')
    regression_net = RegressionNet(net_name, device, order, x_samples, y_samples, runs, batchSize)
    if args.load and os.path.isfile(args.load):
        print(f'Loading model state from: {args.load}')
        state = torch.load(args.load)
        regression_net.load(state)

    if args.eval:
        xVal = float(args.eval)
        t = torch.tensor([xVal])
        predictedVal = regression_net.calc(t)
        print(f'p(x): {xVal} -> {predictedVal}')  # eval the sample x value
        print(f'r(x): {xVal} -> {func(t)}')  # eval the sample x value
    else:
        regression_net.train(writer)  # further train the net

    writer.close()

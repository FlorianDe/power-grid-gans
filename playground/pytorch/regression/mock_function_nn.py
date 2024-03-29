import argparse
import copy
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src.utils.tensorboard_utils import TensorboardUtils, GraphPlotItem


# torch.autograd.set_detect_anomaly(True)


@dataclass
class MinLossModel:
    model: Optional[nn.Module] = None
    step: int = 0
    loss: float = float("inf")
    formula_printed: bool = False


class PolynomialLayer(nn.Module):
    def __init__(self, _order: int, _device):
        super().__init__()
        self.device = _device
        if _order < 1:
            raise ValueError("Order has to be at least 1 for a linear regression!")
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn((), device=self.device), requires_grad=True) for _ in range(_order + 1)])

    def forward(self, x):
        value = torch.zeros(x.size(), requires_grad=False, device=self.device)
        for order, coefficient in enumerate(self.params):
            value += (x.pow(order)).multiply(coefficient)
        return value

    def string(self):
        expressions = [""] * len(self.params)
        for order in range(len(self.params)):
            expressions[order] = f'{self.params[order].item()}' + (f'*x^{order}' if order > 0 else '')
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
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-6, momentum=0.97)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05, betas=(0.5, 0.999))
        self.data_loader = DataLoader(
            TensorDataset(_x_samples, _y_samples),
            batch_size=min(len(_x_samples), _batch_size),
            shuffle=True,
            # num_workers=2
        )
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.97, patience=10, cooldown=10, min_lr=1e-9, threshold=1e-9)

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
        batches = len(self.data_loader)
        min_loss_model = MinLossModel()
        for epoch in tqdm(range(self.runs)):
            running_loss = 0
            for i, data in enumerate(self.data_loader, 0):
                x_samples_batch, y_samples_batch = data
                y_pred = self.model(x_samples_batch)
                loss = self.criterion(y_pred, y_samples_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss

            global_step = epoch * batches
            avg_loss = running_loss / batches
            # Calculate
            self.scheduler.step(avg_loss)
            avg_lr = self.optimizer.param_groups[0]['lr']  # self.scheduler.get_last_lr()[0]
            if avg_loss < min_loss_model.loss:
                min_loss_model.model = copy.deepcopy(self.model)
                min_loss_model.loss = avg_loss
                min_loss_model.step = global_step
                writer.add_scalar(
                    tag='training loss improvements',
                    scalar_value=avg_loss,
                    global_step=global_step
                )

            # only print the formulas after 50% of the training
            if min_loss_model.formula_printed is False and (self.runs*batches * 0.50) < global_step:
                min_loss_model.formula_printed = True
                writer.add_text(net_name, min_loss_model.model.string(), min_loss_model.step)

            if (self.runs / 100) < 2 or (epoch % math.ceil(self.runs / 100)) == 0:
                writer.add_scalar(
                    tag='training loss',
                    scalar_value=avg_loss,
                    global_step=global_step
                )
                writer.add_scalar(
                    tag='learning rate',
                    scalar_value=avg_lr,
                    global_step=global_step
                )
                TensorboardUtils.plot_graph_as_figure(
                    tag="function/comparison",
                    writer=writer,
                    plot_data=[
                        GraphPlotItem(
                            label="real",
                            x=self.x_samples.detach().cpu().numpy(),
                            y=self.y_samples.detach().cpu().numpy(),
                            color='c'
                        ),
                        GraphPlotItem(
                            label="pred",
                            x=self.x_samples.detach().cpu().numpy(),
                            y=self.calc(self.x_samples).detach().cpu().numpy(),
                            color='r'
                        ),
                    ],
                    global_step=global_step
                )

        print(f'Best regression: \n'
              f'step: {min_loss_model.step}\n'
              f'loss: {min_loss_model.loss}\n'
              f'{min_loss_model.model.string()}')
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

    order = 5  # Should be saved too
    start = - math.pi
    end = math.pi
    func = torch.sin
    stepSize = 0.2
    steps = math.floor((end - start) / stepSize)
    batchSize = 10  # steps
    x_samples = torch.linspace(start, end, steps, device=device)
    y_samples = torch.sin(x_samples).to(device)
    runs = 20_000

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

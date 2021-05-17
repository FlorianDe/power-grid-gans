import math
from datetime import datetime

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

# torch.autograd.set_detect_anomaly(True)
from src.tensorboard.utils import TensorboardUtils, GraphPlotItem


class PolynomialLayer(nn.Module):
    def __init__(self, _order: int):
        super().__init__()
        if _order < 1:
            raise ValueError("Order has to be at least 1 for a linear regression!")
        self.params = nn.ParameterList([nn.Parameter(torch.randn(()), requires_grad=True) for _ in range(_order + 1)])

    def forward(self, x):
        value = torch.zeros(x.size(), requires_grad=False)
        for order, coefficient in enumerate(self.params):
            value += (x.pow(order)).multiply(coefficient)
        return value

    def string(self):
        expressions = [""] * len(self.params)
        for order in range(len(self.params)):
            expressions[order] = f'{self.params[order].item()}*x^{order}'
        return 'y=' + ' + '.join(expressions)


class MockFunctionNN:
    def __init__(self, _order: int, _x_samples, _y_samples, _writer, _runs=1, _batch_size=50):
        self.model = PolynomialLayer(_order)
        self.x_samples = _x_samples
        self.y_samples = _y_samples
        self.writer = _writer
        self.runs = _runs
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-6, momentum=0.97)
        self.trainloader = DataLoader(
            TensorDataset(_x_samples, _y_samples),
            batch_size=min(len(_x_samples), _batch_size),
            shuffle=True,
            # num_workers=2
        )

        if torch.cuda.is_available():
            self.x_samples = self.x_samples.to('cuda')
            print("Cuda support available, transferring all tensors to the gpu!")
        else:
            print("No cuda support available!")

    def run(self):
        for epoch in range(self.runs):
            running_loss = 0
            for i, data in enumerate(self.trainloader, 0):
                x_samples_batch, y_samples_batch = data
                y_pred = self.model(x_samples_batch)
                loss = self.criterion(y_pred, y_samples_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss

            if (self.runs / 100) < 2 or (epoch % math.ceil(self.runs / 100)) == 0:
                self.writer.add_scalar(
                    tag='training loss',
                    scalar_value=running_loss/len(self.trainloader),
                    global_step=epoch * len(self.trainloader)
                )
                with torch.no_grad():
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
                                y=self.model(self.x_samples).detach().numpy(),
                                color='r'
                            ),
                        ],
                        global_step=epoch * len(self.trainloader)
                    )

        polynomial_str = self.model.string()
        print(f'Result: {polynomial_str}')
        self.writer.add_text('Result', polynomial_str, runs)


if __name__ == "__main__":
    order = 5
    start = - math.pi
    end = math.pi
    func = torch.sin
    stepSize = 0.2
    steps = math.floor((end - start) / stepSize)
    batchSize = 10  # steps
    x_samples = torch.linspace(start, end, steps)
    y_samples = torch.sin(x_samples)
    runs = 20_000
    writer = SummaryWriter(
        f'runs/regression_order{order:03}_batches{batchSize:04}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    mockFunction = MockFunctionNN(order, x_samples, y_samples, writer, runs, batchSize)
    mockFunction.run()
    writer.close()




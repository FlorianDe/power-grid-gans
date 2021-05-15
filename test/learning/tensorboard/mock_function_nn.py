import math

import torch
import torch.nn as nn

# torch.autograd.set_detect_anomaly(True)

class PolynomialLayer(nn.Module):
    def __init__(self, order: int):
        super().__init__()
        if order < 1:
            raise ValueError("Order has to be at least 1 for a linear regression!")
        self.params = nn.ParameterList([nn.Parameter(torch.randn(()), requires_grad=True) for _ in range(order + 1)])
        print(f'self.params: {self.params}')

    def forward(self, x):
        value = torch.zeros(x.size())
        for i, p in enumerate(self.params):
            value = value + self.params[i] * (x ** i)
            # print(f'value: {value}')
        return value

    def string(self):
        expressions = [""] * len(self.params)
        for order in range(len(self.params)):
            expressions[order] = f'{self.params[order].item()}*x^{order}'
        return 'y=' + ' + '.join(expressions)


class MockFunctionNN:
    def __init__(self, _order: int, _func, _start, _end, _steps):
        self.model = PolynomialLayer(_order)
        self.x_samples = torch.linspace(_start, _end, _steps)
        self.y_real = _func(self.x_samples)
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-6)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-6)

    def run(self):
        for t in range(10_000):
            y_pred = self.model(self.x_samples)
            loss = self.criterion(y_pred, self.y_real)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(f'Result: {self.model.string()}')


if __name__ == "__main__":
    start = - math.pi
    end = math.pi
    steps = 2000

    MockFunctionNN(3, torch.sin, start, end, steps).run()
    # print(polynomialFunction(torch.tensor([1, 2, 3]), [3, 2, 1]))

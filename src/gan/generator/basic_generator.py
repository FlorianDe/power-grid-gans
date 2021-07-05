from typing import Optional

from src.net.dynamic import FNN


class BasicGenerator(FNN):
    def __init__(self, input_size: int, out_size: int, hidden_layers: Optional[list[int]] = None):
        super(BasicGenerator, self).__init__(input_size, out_size, hidden_layers)

    def reshape(self, x):
        return x

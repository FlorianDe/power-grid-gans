from abc import ABC, abstractmethod


class Scaler(ABC):
    @abstractmethod
    def apply(self, t: float) -> float:
        raise NotImplementedError("Please Implement this method")

    def __call__(self, t: float):
        return self.apply(t)


class IntervalScaler(Scaler, ABC):
    def __init__(self, source: (float, float), destination: (float, float)) -> None:
        super().__init__()
        self.source = source
        self.destination = destination


class LinearIntervalScaler(IntervalScaler):
    def apply(self, t: float) -> float:
        a = self.source[0]
        b = self.source[1]
        c = self.destination[0]
        d = self.destination[1]
        return c + ((d - c) / (b - a)) * (t - a)


if __name__ == '__main__':
    scaler = LinearIntervalScaler(source=(-5, 10), destination=(-20, 5))
    print(scaler(-5))
    print(scaler(10))

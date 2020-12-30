import numpy as np

from function import Function


class ReLU(Function):
    inplace: bool
    activated: np.ndarray

    def __init__(self, inplace=True):
        self.type = 'activation'
        self.inplace = inplace

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.inplace:
            x[x < 0] = 0.
            self.activated = x
        else:
            self.activated = x * (x > 0)
        return self.activated

    def backward(self, d_y: np.ndarray) -> np.ndarray:
        return d_y * (self.activated > 0)


class Tanh(Function):
    inplace: bool
    activated: np.ndarray

    def __init__(self, inplace=True):
        self.type = 'activation'
        self.inplace = inplace

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.activated = np.tanh(x)
        return self.activated

    def backward(self, d_y: np.ndarray) -> np.ndarray:
        return 1.0 - np.tanh(d_y) ** 2

import numpy as np


class Tensor:
    def __init__(self, shape):
        self.data = np.ndarray(shape, np.float32)
        self.grad = np.ndarray(shape, np.float32)


class Function(object):
    weights: Tensor
    bias: Tensor
    type: str
    input: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, d_y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def getParams(self):
        return [self.weights, self.bias]

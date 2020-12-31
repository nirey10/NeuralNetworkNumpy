import numpy as np


class Abstract_Activation(object):
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, d_y: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ReLU(Abstract_Activation):
    activation_output: np.ndarray

    def __init__(self):
        self.type = 'activation'
        self.activation_output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x[x < 0] = 0.
        self.activation_output = x
        return self.activation_output

    def backward(self, d_y: np.ndarray) -> np.ndarray:
        return d_y * (self.activation_output > 0)


class Tanh(Abstract_Activation):
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

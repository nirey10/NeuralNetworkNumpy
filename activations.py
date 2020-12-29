import numpy as np

class Function(object):
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def getParams(self):
        return []

class ReLU(Function):
    def __init__(self, inplace=True):
        self.type = 'activation'
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x[x < 0] = 0.
            self.activated = x
        else:
            self.activated = x * (x > 0)

        return self.activated

    def backward(self, d_y):
        return d_y * (self.activated > 0)


class Tanh(Function):
    def __init__(self, inplace=True):
        self.type = 'activation'
        self.inplace = inplace

    def forward(self, x):
        self.activated = np.tanh(x)

        return self.activated

    def backward(self, d_y):
        return 1.0 - np.tanh(d_y) ** 2

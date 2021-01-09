import numpy as np

from layers import Softmax
from utils import get_data, cross_entropy_loss
import matplotlib.pyplot as plt


X, Y, _, _ = get_data('PeaksData')

soft_max_layer = Softmax(2, 5)

x, y = X[0], Y[0]


def f(xx):
    return soft_max_layer.forward(xx)


def grad_w(xx):
    error = (f(xx)[range(len(y)), y] - 1) / len(y)
    return np.dot(xx.T, error)


def grad_b(xx):
    error = (f(xx)[range(len(y)), y] - 1) / len(y)
    return np.sum(error, axis=0, keepdims=True)


d = np.random.random(len(x))

eps0 = 1

eps = np.array([(0.5**i)*eps0 for i in range(10)])

# plt.plot(eps, [np.abs(f(x+epss*d) - f(x)) for epss in eps])
# plt.show()

print(x)
print(f(X))
print(cross_entropy_loss(f(X), Y))

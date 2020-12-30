from activations import Abstract_Activation
import numpy as np
import utils

class Abstract_Model(object):
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

class MyNeuralNetwork(Abstract_Model):
    def __init__(self):
        self.graph = []
        self.parameters = []

    def add(self, layer):
        self.graph.append(layer)
        if not isinstance(layer, Abstract_Activation):  # in case of adding an activation
            self.parameters += layer.parameters()

    def init(self):
        for f in self.graph:
            if f.type == 'linear':
                weights, bias = f.parameters()
                weights.data = .01 * np.random.randn(weights.data.shape[0], weights.data.shape[1])
                bias.data = 0.


    def fit(self, data, target, batch_size, num_epochs, optimizer):
        loss_history = []
        self.init()
        data_gen = utils.DataGenerator(data, target, batch_size)
        itr = 0
        for epoch in range(num_epochs):
            for X, Y in data_gen:
                optimizer.zeroGrad()
                #for f in self.graph: X = f.forward(X)
                probabilities = self.forward(X)
                loss = utils.cross_entropy_loss(probabilities, Y)
                self.backward(Y)
                loss_history += [loss]
                print("Loss at epoch = {} and iteration = {}: {}".format(epoch, itr, loss_history[-1]))
                itr += 1
                optimizer.step()

        return loss_history

    def forward(self, X):
        for f in self.graph: X = f.forward(X)
        return X

    def backward(self, true_label):
        grad = true_label
        for f in self.graph[::-1]: grad = f.backward(grad)

    def predict(self, data):
        X = data
        for f in self.graph: X = f.forward(X)
        return X
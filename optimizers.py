from network_tests import grad_test
class Optimizer(object):
    def __init__(self, parameters):
        self.parameters = parameters

    def step(self):
        raise NotImplementedError

    def fit(self, data, target, batch_size, epochs):
        raise NotImplementedError

    def zeroGrad(self):
        for p in self.parameters:
            p.grad = 0.


class SGD(Optimizer):
    def __init__(self, parameters, lr=.001):
        super().__init__(parameters)
        self.lr = lr

    def step(self):

        for p in self.parameters:
            p.data = p.data - self.lr * p.grad

    def fit(self, data, target, batch_size, epochs):
        # loss_history = []
        # data_gen = utils.DataGenerator(data, target, batch_size)
        # iteration = 0
        # for epoch in range(epochs):
        #     for train_X, train_Y in data_gen:
        #         self.zeroGrad()
        #         probabilities = self.model.forward(train_X)
        #         loss = utils.cross_entropy_loss(probabilities, train_Y)
        #         self.model.backward(train_Y)
        #         loss_history += [loss]
        #         print("Loss at epoch = {} and iteration = {}: {}".format(epoch, iteration, loss_history[-1]))
        #         iteration += 1
        #         self.step()
        # return loss_history
        pass

# class SGD(Optimizer):
#     def __init__(self, parameters, lr=.001, weight_decay=0.0, momentum=.9):
#         super().__init__(parameters)
#         self.lr = lr
#         self.weight_decay = weight_decay
#         self.momentum     = momentum
#         self.velocity     = []
#         for p in parameters:
#             self.velocity.append(np.zeros_like(p.grad))
#
#     def step(self):
#         for p,v in zip(self.parameters,self.velocity):
#             v = self.momentum*v+p.grad+self.weight_decay*p.data
#             p.data=p.data-self.lr*v

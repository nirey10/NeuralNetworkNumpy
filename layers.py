import numpy as np


class Tensor():
    def __init__(self, shape):
        self.data = np.ndarray(shape, np.float32)
        self.grad = np.ndarray(shape, np.float32)


class Function(object):
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def getParams(self):
        return []


class Linear(Function):
    def __init__(self,in_nodes,out_nodes):
        self.weights = Tensor((in_nodes,out_nodes))
        self.bias    = Tensor((1,out_nodes))
        self.type = 'linear'

    def forward(self,x):
        output = np.dot(x,self.weights.data)+self.bias.data
        self.input = x
        return output

    def backward(self,d_y):
        self.weights.grad += np.dot(self.input.T,d_y)
        self.bias.grad    += np.sum(d_y,axis=0,keepdims=True)
        grad_input         = np.dot(d_y,self.weights.data.T)
        return grad_input

    def getParams(self):
        return [self.weights,self.bias]

class Softmax(Function):
    def __init__(self,in_nodes,out_nodes):
        self.weights = Tensor((in_nodes,out_nodes))
        self.bias    = Tensor((1,out_nodes))
        self.type = 'softmax'

    def forward(self,x):
        output = np.dot(x,self.weights.data)+self.bias.data
        unnormalized_proba = np.exp(output - np.max(output, axis=1, keepdims=True))
        self.proba = unnormalized_proba / np.sum(unnormalized_proba, axis=1, keepdims=True)
        self.input = x
        return self.proba

    def backward(self,target):
        self.target = target
        d_y = self.proba
        d_y[range(len(self.target)), self.target] -= 1.0
        d_y /= len(self.target)
        self.weights.grad += np.dot(self.input.T,d_y)
        self.bias.grad    += np.sum(d_y,axis=0,keepdims=True)
        grad_input         = np.dot(d_y,self.weights.data.T)
        return grad_input

    def getParams(self):
        return [self.weights,self.bias]

class SoftmaxWithLoss(Function):
    def __init__(self):
        self.type = 'normalization'

    def forward(self,x):
        unnormalized_proba = np.exp(x-np.max(x,axis=1,keepdims=True))
        self.proba         = unnormalized_proba/np.sum(unnormalized_proba,axis=1,keepdims=True)
        #loss               = -np.log(self.proba[range(len(target)),target])
        return self.proba

    def backward(self, target):
        self.target = target
        gradient = self.proba
        gradient[range(len(self.target)),self.target]-=1.0
        gradient/=len(self.target)
        return gradient
import numpy as np
import matplotlib.pyplot as plt
import models
import layers
import copy
import utils
import activations

def grad_test(X_train, y_train):
    softmax_in = 2
    softmax_out = 5
    model = models.MyNeuralNetwork()
    model.add(layers.Softmax(softmax_in, softmax_out))
    model.init()
    for p in model.parameters:
        p.grad = 0.

    eps0 = 1
    eps = np.array([(0.5 ** i) * eps0 for i in range(10)])

    one_d = np.random.randn(X_train.shape[1])
    d = np.tile(one_d, (X_train.shape[0], 1))

    grad_diff = []

    for epss in eps:
        X_train_test = X_train + epss * d
        model_grad = copy.deepcopy(model)
        probabilities_grad = model_grad.forward(np.array(X_train))
        model2 = copy.deepcopy(model)
        probabilities_grad2 = model2.forward(np.array(X_train_test))

        grad_diff.append(np.abs(utils.cross_entropy_loss(probabilities_grad2, np.array([1])) - utils.cross_entropy_loss(probabilities_grad, np.array([1]))))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle('Gradient test for softmax by weights', fontsize=16)

    axs[0, 0].plot(eps, grad_diff)
    axs[0, 0].set_xlabel('$\epsilon$')
    axs[0, 0].set_title('$|f(x+\epsilon d) - f(x)|$')

    axs[0, 1].plot(range(len(grad_diff) - 1), [grad_diff[i + 1] / grad_diff[i] for i in range(len(grad_diff) - 1)])
    axs[0, 1].set_xlabel('$i$')
    axs[0, 1].set_title('rate of decrease')
    axs[0, 1].set_ylim([0, 1])

    grad_diff = []
    for epss in eps:
        X_train_test = X_train + epss * d
        model_grad = copy.deepcopy(model)
        probabilities_grad = model_grad.forward(np.array(X_train))
        model2 = copy.deepcopy(model)
        probabilities_grad2 = copy.deepcopy(model2.forward(np.array(X_train_test)))
        model2.backward(y_train)
        grad_x = model2.graph[0].weights.grad
        grad_diff.append(np.abs(utils.cross_entropy_loss(probabilities_grad2, y_train) -
                                utils.cross_entropy_loss(probabilities_grad, y_train) -
                                epss * sum(np.dot(np.array([one_d]) ,grad_x).T)))
    # np.dot(d.T, grad_x)))
    axs[1, 0].plot(eps, grad_diff)
    axs[1, 0].set_xlabel('$\epsilon$')
    axs[1, 0].set_title('$|f(x+\epsilon d) - f(x) - \epsilon d^{T} grad(x)|$')

    axs[1, 1].plot(range(len(grad_diff) - 1), [grad_diff[i + 1] / grad_diff[i] for i in range(len(grad_diff) - 1)])
    axs[1, 1].set_xlabel('$i$')
    axs[1, 1].set_title('rate of decrease')
    axs[1, 1].set_ylim([0, 1])

    plt.show()

def Jac_MV(model):
    jacobian = model.graph[0].weights.grad
    for layer in model.graph[1:]:
        if layer.type == 'activation':
            continue
        jacobian = np.dot(jacobian, layer.weights.grad)

    return jacobian

def jacobian_test(X_train, y_train):

    softmax_in = 2
    softmax_out = 5
    hidden_units = 10
    model = models.MyNeuralNetwork()
    model.add(layers.Linear(softmax_in, hidden_units))
    model.add(activations.Tanh())
    model.add(layers.Softmax(hidden_units, softmax_out))
    model.init()
    for p in model.parameters:
        p.grad = 0.

    eps0 = 1
    eps = np.array([(0.5 ** i) * eps0 for i in range(10)])

    one_d = np.random.randn(X_train.shape[1])
    d = np.tile(one_d, (X_train.shape[0], 1))

    grad_diff = []

    for epss in eps:
        X_train_test = X_train + epss * d
        model_grad = copy.deepcopy(model)
        probabilities_grad = model_grad.forward(np.array(X_train))
        model2 = copy.deepcopy(model)
        probabilities_grad2 = model2.forward(np.array(X_train_test))

        grad_diff.append(np.abs(utils.cross_entropy_loss(probabilities_grad2, np.array([1])) - utils.cross_entropy_loss(probabilities_grad, np.array([1]))))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle('Jacobian test for softmax by weights', fontsize=16)

    axs[0, 0].plot(eps, grad_diff)
    axs[0, 0].set_xlabel('$\epsilon$')
    axs[0, 0].set_title('$|f(x+\epsilon d) - f(x)|$')

    axs[0, 1].plot(range(len(grad_diff) - 1), [grad_diff[i + 1] / grad_diff[i] for i in range(len(grad_diff) - 1)])
    axs[0, 1].set_xlabel('$i$')
    axs[0, 1].set_title('rate of decrease')
    axs[0, 1].set_ylim([0, 1])

    grad_diff = []
    for epss in eps:
        X_train_test = X_train + epss * d
        model_grad = copy.deepcopy(model)
        probabilities_grad = model_grad.forward(np.array(X_train))
        model2 = copy.deepcopy(model)
        probabilities_grad2 = copy.deepcopy(model2.forward(np.array(X_train_test)))
        model2.backward(y_train)
        jacobian = Jac_MV(model2)
        #grad_x = model2.graph[0].weights.grad
        grad_diff.append(np.abs(utils.cross_entropy_loss(probabilities_grad2, y_train) -
                                utils.cross_entropy_loss(probabilities_grad, y_train) -
                                epss * sum(np.dot(np.array([one_d]), jacobian).T)))
    # np.dot(d.T, grad_x)))
    axs[1, 0].plot(eps, grad_diff)
    axs[1, 0].set_xlabel('$\epsilon$')
    axs[1, 0].set_title('$|f(x+\epsilon d) - f(x) - \epsilon d^{T} grad(x)|$')


    axs[1, 1].plot(range(len(grad_diff) - 1), [grad_diff[i + 1] / grad_diff[i] for i in range(len(grad_diff) - 1)])
    axs[1, 1].set_xlabel('$i$')
    axs[1, 1].set_title('rate of decrease')
    axs[1, 1].set_ylim([0, 1])

    plt.show()
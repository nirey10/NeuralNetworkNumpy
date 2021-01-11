import numpy as np
import matplotlib.pyplot as plt
import models
import layers
import copy
import utils
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

    d = np.random.random(model.parameters[0].data.ravel().shape)
    d = d / np.sum(d)
    reshaped_d = d.reshape((softmax_in, softmax_out))
    label = np.array([[1, 0, 0, 0, 0]])
    grad_diff = []

    for epss in eps:
        model_grad = copy.deepcopy(model)
        probabilities_grad = model_grad.forward(np.array([X_train[0]]))
        model2 = copy.deepcopy(model)
        model2.graph[0].weights.data += epss * reshaped_d
        probabilities_grad2 = model2.forward(np.array([X_train[0]]))

        grad_diff.append(np.abs(utils.cross_entropy_loss(probabilities_grad2, np.array([1])) - utils.cross_entropy_loss(probabilities_grad, np.array([1]))))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle('Gradient test for softmax by weights', fontsize=16)

    axs[0, 0].plot(eps, grad_diff)
    axs[0, 0].set_xlabel('$\epsilon$')
    axs[0, 0].set_title('$|f(x+\epsilon d) - f(x)|$')

    # axs[0, 1].plot(range(len(res1) - 1), [res1[i + 1] / res1[i] for i in range(len(res) - 1)])
    # axs[0, 1].set_xlabel('$i$')
    # axs[0, 1].set_title('rate of decrease')
    # axs[0, 1].set_ylim([0, 1])
    #
    # res2 = [np.abs(f(w + epss * d.reshape((2, 5)), b) - f(w, b) - epss * np.dot(d.T, grad_w(w, b))) for epss in eps]
    #
    X = X_train[0:100]
    Y = y_train[0:100]
    grad_diff = []
    for epss in eps:
        model_grad = copy.deepcopy(model)
        probabilities_grad = model_grad.forward(np.array(X))
        model2 = copy.deepcopy(model)
        #model2.graph[0].weights.data += epss * reshaped_d
        probabilities_grad2 = copy.deepcopy(model2.forward(np.array(X)))
        model2.backward(Y)
        grad_x = model2.graph[0].weights.grad.reshape((softmax_in * softmax_out))
        grad_diff.append(np.abs(utils.cross_entropy_loss(probabilities_grad2, Y) -
                                utils.cross_entropy_loss(probabilities_grad, Y) -
                                epss * np.dot(d.T, grad_x)))

    axs[1, 0].plot(eps, grad_diff)
    axs[1, 0].set_xlabel('$\epsilon$')
    axs[1, 0].set_title('$|f(x+\epsilon d) - f(x) - \epsilon d^{T} grad(x)|$')

    #
    # axs[1, 1].plot(range(len(res2) - 1), [res2[i + 1] / res2[i] for i in range(len(res) - 1)])
    # axs[1, 1].set_xlabel('$i$')
    # axs[1, 1].set_title('rate of decrease')
    # axs[1, 1].set_ylim([0, 1])

    plt.show()
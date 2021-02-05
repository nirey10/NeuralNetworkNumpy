import utils
import numpy as np
from sklearn.utils import shuffle
import activations
import layers
import optimizers
import models
from network_tests import grad_test_W, grad_test_b, jacobian_test_W, jacobian_test_b
# TODO Organize this file

if __name__=="__main__":

    batch_size  = 50
    num_epochs = 200
    samples_per_class = 10000
    num_classes = 2
    hidden_units = 100
    hidden_units2 = 10
    dimensions = 2
    # num_accuracy_calc = 1000  # number of samples to take for the accuracy plots

    # PeaksData  da, SwissRollData, GMMData
    X_train, y_train, X_test, y_test = utils.get_data('PeaksData')
    X_train, y_train = shuffle(X_train, y_train)

    grad_test_W(X_train, y_train)
    grad_test_b(X_train, y_train)
    jacobian_test_W(X_train, y_train)
    jacobian_test_b(X_train, y_train)

    model = models.MyNeuralNetwork()
    model.add(layers.Linear(dimensions, hidden_units))
    model.add(activations.ReLU())
    model.add(layers.Softmax(hidden_units, 5))
    optimizer = optimizers.SGD(model.parameters, lr=0.1)
    losses, train_accuracy, test_accuracy = model.fit(X_train, y_train, X_test, y_test, batch_size, num_epochs, optimizer)


    # plotting
    utils.plot_scores(train_accuracy, test_accuracy)
    utils.plot2DDataWithDecisionBoundary(X_test, y_test, model)



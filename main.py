import utils
import numpy as np
from sklearn.utils import shuffle
import activations
import layers
import optimizers
import models

# TODO Organize this file

if __name__=="__main__":

    batch_size  = 5
    num_epochs = 10
    samples_per_class = 10000
    num_classes = 2
    hidden_units = 100
    hidden_units2 = 10
    dimensions = 2
    # num_accuracy_calc = 1000  # number of samples to take for the accuracy plots

    # PeaksData, SwissRollData, GMMData
    X_train, y_train, X_test, y_test = utils.get_data('SwissRollData')
    X_train, y_train = shuffle(X_train, y_train)
    #X_train, y_train = utils.genSpiralData(samples_per_class, num_classes)

    model = models.MyNeuralNetwork()
    model.add(layers.Linear(dimensions, hidden_units))
    model.add(activations.ReLU())
    model.add(layers.Softmax(hidden_units, num_classes))
    optimizer = optimizers.SGD(model.parameters, lr=0.1)
    losses, train_accuracy, test_accuracy = model.fit(X_train, y_train, X_test, y_test, batch_size, num_epochs, optimizer, num_accuracy_calc)

    # plotting
    utils.plot_scores(train_accuracy)
    utils.plot_scores(test_accuracy)
    utils.plot2DDataWithDecisionBoundary(X_test, y_test, model)



import utils
import numpy as np
from sklearn.utils import shuffle
import activations
import layers
import optimizers
import models

# TODO Organize this file

if __name__=="__main__":

    batch_size  = 50
    num_epochs = 100
    samples_per_class = 10000
    num_classes = 5
    hidden_units = 100
    hidden_units2 = 10
    dimensions = 2

    X_train, X_test, y_train, y_test = utils.get_data('PeaksData')
    data, target = shuffle(X_train, y_train)
    data, target = utils.genSpiralData(samples_per_class, num_classes)

    model = models.MyNeuralNetwork()
    model.add(layers.Linear(dimensions, hidden_units))
    model.add(activations.ReLU())
    model.add(layers.Softmax(hidden_units, num_classes))
    optimizer = optimizers.SGD(model.parameters, lr=0.1)
    model.fit(data, target, batch_size, num_epochs, optimizer)

    # plotting
    predicted_labels = np.argmax(model.predict(data), axis=1)
    accuracy = np.sum(predicted_labels == target)/len(target)
    print("Model Accuracy = {}".format(accuracy))
    utils.plot2DDataWithDecisionBoundary(data, target, model)



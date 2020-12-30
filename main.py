import utilities
import numpy as np
import scipy.io
from sklearn.utils import shuffle
import activations
import layers
import optimizers

# TODO Organize this file

if __name__=="__main__":
    SwissRoll = scipy.io.loadmat('dataset/SwissRollData.mat')
    y_train = np.array(SwissRoll['Ct'])
    X_train = np.array(SwissRoll['Yt'])
    y_test = np.array(SwissRoll['Cv'])
    X_test = np.array(SwissRoll['Yv'])

    X_train = np.transpose(X_train)
    y_train = np.transpose(y_train)
    X_test = np.transpose(X_test)
    y_test = np.transpose(y_test)

    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    #X_train, y_train = shuffle(X_train, y_train)
    #X_train = np.sin(X_train)

    batch_size        = 50
    num_epochs        = 100
    samples_per_class = 10000
    num_classes       = 2
    hidden_units      = 100
    hidden_units2 =    10
    data,target       = shuffle(X_train, y_train)
    data,target       =utilities.genSpiralData(samples_per_class,num_classes)
    model             = utilities.Model()
    model.add(layers.Linear(2,hidden_units))
    model.add(activations.ReLU())
    model.add(layers.Softmax(hidden_units,num_classes))
    optim   = optimizers.SGD(model.parameters,lr=0.1)
    loss_fn = layers.SoftmaxWithLoss()
    model.fit(data,target,batch_size,num_epochs,optim,loss_fn)
    predicted_labels = np.argmax(model.predict(data),axis=1)
    accuracy         = np.sum(predicted_labels==target)/len(target)
    print("Model Accuracy = {}".format(accuracy))
    utilities.plot2DDataWithDecisionBoundary(data,target,model)



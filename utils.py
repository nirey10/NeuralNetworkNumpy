import numpy as np
import matplotlib.pyplot as plt


def cross_entropy_loss(X,target):
    probability = X
    loss = -np.log(probability[range(len(target)), target])
    return loss.mean()


def plot2DData(data,target):
    plt.scatter(x = data[:,0],y = data[:,1],c = target,cmap=plt.cm.rainbow)
    plt.show()


def plot2DDataWithDecisionBoundary(data,target,model):
    x_min,x_max = np.min(data[:,0])-.5,np.max(data[:,0])+.5
    y_min,y_max = np.min(data[:,1])-.5,np.max(data[:,1])+.5
    X,Y = np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02)
    XX,YY = np.meshgrid(X,Y)
    Z = np.argmax(model.predict(np.c_[XX.ravel(),YY.ravel()]),axis=1).reshape(XX.shape)
    plt.contourf(XX,YY,Z,cmap=plt.cm.seismic)
    plt.scatter(x=data[:,0],y=data[:,1],c=target,cmap=plt.cm.seismic)
    plt.show()


def genSpiralData(points_per_class,num_classes):  # TODO: REMOVE AFTER TESTING
    data   = np.ndarray((points_per_class*num_classes,2),np.float32)
    target = np.ndarray((points_per_class*num_classes,),np.uint8)
    r = np.linspace(0,1,points_per_class)
    radians_per_class = 2*np.pi/num_classes
    for i in range(num_classes):
        t = np.linspace(i*radians_per_class,(i+1.5)*radians_per_class,points_per_class)+0.1*np.random.randn(points_per_class)
        data[i*points_per_class:(i+1)*points_per_class] = np.c_[r*np.sin(t),r*np.cos(t)]
        target[i*points_per_class:(i+1)*points_per_class] = i
    return  data,target


class DataGenerator():  # TODO: IMPLEMENT LIKE A NOOB IN THE FOR LOOP, DONT USE THIS NICE THING
    def __init__(self, data, target, batch_size, shuffle=True):
        self.shuffle      = shuffle
        if shuffle:
            shuffled_indices = np.random.permutation(len(data))
        else:
            shuffled_indices = range(len(data))

        self.data         = data[shuffled_indices]
        self.target       = target[shuffled_indices]
        self.batch_size   = batch_size 
        self.num_batches  = int(np.ceil(data.shape[0]/batch_size))
        self.counter      = 0


    def __iter__(self):
        return self

    def __next__(self):
        if self.counter<self.num_batches:
            batch_data = self.data[self.counter*self.batch_size:(self.counter+1)*self.batch_size]
            batch_target = self.target[self.counter*self.batch_size:(self.counter+1)*self.batch_size]
            self.counter+=1
            return batch_data,batch_target
        else:
            if self.shuffle:
                shuffled_indices = np.random.permutation(len(self.target))
            else:
                shuffled_indices = range(len(self.target))

            self.data         = self.data[shuffled_indices]
            self.target       = self.target[shuffled_indices]

            self.counter = 0
            raise StopIteration










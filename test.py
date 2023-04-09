from dataset.load_MNIST import MNIST
from model.Model import *
import pickle
def cnn_mnist():
    mnist = MNIST(dimension=3)
    model = Model(name='model', input_dim=[28, 28, 1])
    model.initial(
        [
            Conv2D(name='C1', kernel_size=[3, 3], filters=5, padding='valid'),
            Activation(name='A1', method='relu'),
            MaxPooling2D(name='P1', pooling_size=[2, 2]),
            Flatten(name='flatten'),
            Dense(name='fc1', units=100),
            Activation(name='A3', method='relu'),
            Dense(name='fc2', units=10),
            Activation(name='A4', method='softmax'),
        ]
    )
    
    model.load_model("./model/")
    print('Test_Acc=[{}]'.format(model.measure(mnist.test_x, mnist.test_y)))
if __name__ == '__main__':

    cnn_mnist()
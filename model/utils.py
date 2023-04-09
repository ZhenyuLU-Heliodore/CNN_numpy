import numpy as np
#this code is for basic math process
#2023/4/9


def sigmoid(x_set):
    return 1 / (1 + np.exp(-x_set))


def d_sigmoid(x_set):
    return (1 - sigmoid(x_set)) * sigmoid(x_set)


def softmax(x_set):
    x_row_max = x_set.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x_set.shape)[:-1] + [1])
    x_set = x_set - x_row_max
    x_exp = np.exp(x_set)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x_set.shape)[:-1] + [1])
    return x_exp / x_exp_row_sum


def relu(x_set):
    return np.maximum(0, x_set)


def d_relu(x_set):
    x_set[x_set <= 0] = 0
    x_set[x_set > 0] = 1
    return x_set


def tanh(x_set):
    return np.tanh(x_set)


def d_tanh(x_set):
    return 1 - np.tanh(x_set) ** 2


def activation_function(method, x):
    if method == 'relu':
        a = relu(x)
    elif method == 'sigmoid':
        a = sigmoid(x)
    elif method == 'softmax':
        a = softmax(x)
    elif method is None:
        a = x
    else:
        a = []
        print("No such activation: {}!".format(method))
        exit(1)
    return a


def derivative_function(method, x):
    if method == 'relu':
        d = d_relu(x)
    elif method == 'sigmoid':
        d = d_sigmoid(x)
    elif method is None:
        d = 1
    else:
        d = []
        print("No such activation: {}!".format(method))
        exit(1)
    return d

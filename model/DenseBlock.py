
import numpy as np
# dense block for nn to conect
# forward:y=wx+b
# backward:cal the w size

class Dense(object):
    def __init__(self, name, units):
        self.name = name
        self.__units = units
        self.__input_dim = None
        self.__w = None
        self.__b = None

    def initial(self, input_dim):
        if len(input_dim) != 1:
            print("{} initial error!".format(self.name))
            exit(1)
        self.__input_dim = input_dim
        std = np.sqrt(2. / (self.__input_dim[0]))  # He normalization
        self.__w = np.random.normal(loc=0., scale=std, size=[self.__input_dim[0], self.__units])
        self.__b = np.random.normal(loc=0., scale=std, size=[self.__units])
        return self.name, [self.__units]

    def weight_shape(self):
        return {'w': self.__w.shape, 'b': self.__b.shape}

    def forward(self, _x_set):
        if list(_x_set.shape[1:]) != list(self.__input_dim):
            print("{} input set dim error!".format(self.name))
            exit(1)
        _z = np.dot(_x_set, self.__w) + self.__b
        return _z

    def backward(self, _e_set):
        _e_down_set = np.dot(_e_set, self.__w.transpose())
        return _e_down_set

    def gradient(self, _z_down_set, _e_set):
        _e_set = _e_set.copy()
        nums = len(_z_down_set)
        _z_down_set_m1 = np.expand_dims(_z_down_set, 2)
        _e_set_1n = np.expand_dims(_e_set, 1)
        _dw = np.matmul(_z_down_set_m1, _e_set_1n)
        _dw = np.sum(_dw, axis=0) / nums
        _db = _e_set
        _db = np.sum(_db, axis=0) / nums
        return {'w': _dw, 'b': _db}

    def gradient_descent(self, _g, test_lr=1.):
        _dw = _g['w']
        _db = _g['b']
        self.__w -= test_lr * _dw
        self.__b -= test_lr * _db




import numpy as np
from model.utils import *
#for our way to activate relu

class Activation(object):
    def __init__(self, name, method):
        self.name = name
        self.__method = method
        self.__input_dim = None

    def initial(self, input_dim):
        self.__input_dim = input_dim
        return self.name, self.__input_dim

    def forward(self, _x_set):
        _a_set = activation_function(self.__method, _x_set)
        return _a_set

    def backward(self, _e_set, _z_down_set):
        _e_down_set = derivative_function(self.__method, _z_down_set) * _e_set
        return _e_down_set




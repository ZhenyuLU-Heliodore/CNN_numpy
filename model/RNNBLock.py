
import numpy as np
from model.utils import *


class BasicRNN(object):
    """unidirection, static"""
    def __init__(self, name, units, return_last_step):
        self.name = name
        self.units = units
        self.__return_last_step = return_last_step

        self.u = None
        self.w = None
        self.b = None

        self.h_set = None
        self.s_set = None

        self.input_dim = None
        self.step = None
        self.output_dim = None

    def initial(self, input_dim):
        self.input_dim = input_dim  # [step, inputs]
        self.step = self.input_dim[0]

        std = np.sqrt(1. / self.units)  # Xavier initialization
        self.u = np.random.normal(loc=0., scale=std, size=[self.units, self.units])
        self.b = np.random.normal(loc=0., scale=std, size=[self.units])
        std = np.sqrt(1. / self.input_dim[1])  # Xavier initialization
        self.w = np.random.normal(loc=0., scale=std, size=[self.input_dim[1], self.units])

        self.output_dim = [self.units] if self.__return_last_step else [self.step, self.units]

        return self.name, self.output_dim

    def weight_shape(self):
        return {'u': self.u.shape, 'w': self.w.shape, 'b': self.b.shape}

    def forward(self, _x_set):
        if list(_x_set.shape[1:]) != list(self.input_dim):
            print("{} input set dim error!".format(self.name))
            exit(1)
        shape = _x_set.shape  # [nums, step, inputs]
        nums = shape[0]
        # _x_set = _x_set.cppy()
        _x_set = _x_set.transpose([1, 0, 2])  # [step, nums, inputs]
        _h = np.zeros([self.step + 1, nums, self.units])  # [step+1, nums, units] & zero initial state
        _s = np.zeros([self.step, nums, self.units])
        for t in range(self.step):
            t_h = t + 1
            _s[t] = np.dot(_h[t_h - 1], self.u) + np.dot(_x_set[t], self.w) + self.b
            _h[t_h] = tanh(_s[t])
        _z = _h[-1, :, :] if self.__return_last_step else _h[1:, :, :].transpose([1, 0, 2])
        self.h_set = _h.transpose([1, 0, 2])
        self.s_set = _s.transpose([1, 0, 2])
        return _z

    def backward(self, _e_set):
        _e_set = _e_set.copy()  # [nums, units] or [nums, step, units]
        nums = _e_set.shape[0]
        if len(_e_set.shape) == 2:  # [nums, units]
            _e_set_temp = np.zeros([self.step, nums, self.units])  # [step, nums, units]
            _e_set_temp[self.step - 1] = _e_set
            _e_set = _e_set_temp  # [step, nums, units]
        else:   # [nums, step, units]
            _e_set = _e_set.transpose([1, 0, 2])    # [step, nums, units]
        _h = self.h_set.transpose([1, 0, 2])  # [step+1, nums, units]
        _e_down_t_set = np.zeros([self.step, nums, self.input_dim[1]])  # [step, nums, inputs]
        for t in range(self.step):
            t_h = t + 1
            _e_k_set = np.zeros([t + 1, nums, self.units])
            _e_k_set[t] = np.multiply((1 - _h[t_h] ** 2), _e_set[t])
            for k in range(t - 1, -1, -1):
                k_h = k + 1
                _e_k_set[k] = np.multiply((1 - _h[k_h] ** 2), np.dot(_e_k_set[k + 1], self.u.transpose()))
                _e_down_t_set[k] += np.dot(_e_k_set[k], self.w.transpose())
        _e_down_set = _e_down_t_set.transpose([1, 0, 2])  # [nums, step, inputs]
        return _e_down_set

    def gradient(self, _z_down_set, _e_set):
        _z_down_set = _z_down_set.copy()  # [nums, step, inputs]
        _e_set = _e_set.copy()  # [nums, units] or [nums, step, units]
        nums = len(_e_set)
        if len(_e_set.shape) == 2:  # [nums, units]
            _e_set_temp = np.zeros([self.step, nums, self.units])  # [step, nums, units]
            _e_set_temp[self.step - 1] = _e_set
            _e_set = _e_set_temp  # [step, nums, units]
        else:   # [nums, step, units]
            _e_set = _e_set.transpose([1, 0, 2])    # [step, nums, units]
        _h = self.h_set.transpose([1, 0, 2])  # [step+1, nums, units]
        _x = _z_down_set.transpose([1, 0, 2])    # [step, nums, units]
        _du_t = np.zeros([self.step, nums, self.units, self.units])
        _dw_t = np.zeros([self.step, nums, self.input_dim[1], self.units])
        _db_t = np.zeros([self.step, nums, self.units])
        for t in range(self.step):
            t_h = t + 1
            _e_k_set = np.zeros([t + 1, nums, self.units])
            _e_k_set[t] = np.multiply((1 - _h[t_h] ** 2), _e_set[t])
            for k in range(t-1, -1, -1):
                k_h = k + 1
                _e_k_set[k] = np.multiply((1 - _h[k_h] ** 2), np.dot(_e_k_set[k+1], self.u.transpose()))
                _du_t[t] += np.matmul(np.expand_dims(_h[k_h - 1], -1), np.expand_dims(_e_k_set[k], -2))
                _dw_t[t] += np.matmul(np.expand_dims(_x[k], -1), np.expand_dims(_e_k_set[k], -2))
                _db_t[t] += _e_k_set[k]
        _du = np.sum(_du_t, axis=(0, 1)) / nums  # [step, nums, units, units] --> [units, units] / nums
        _dw = np.sum(_dw_t, axis=(0, 1)) / nums  # [inputs, units] / nums
        _db = np.sum(_db_t, axis=(0, 1)) / nums  # [units] / nums
        return {'w': _dw, 'u': _du, 'b': _db}

    def gradient_descent(self, _g, test_lr=1.):
        _du = _g['u']
        _dw = _g['w']
        _db = _g['b']
        self.u -= test_lr * _du
        self.w -= test_lr * _dw
        self.b -= test_lr * _db





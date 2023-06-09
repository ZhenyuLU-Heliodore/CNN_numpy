
import numpy as np
from model.ConvBlock import Conv2D
from model.PoolingBlock import MaxPooling2D
from model.FlattenBlock import Flatten
from model.DenseBlock import Dense
from model.ActivationBlock import Activation
import pickle
import matplotlib.pyplot as plt
#Model to train and test
#and have Dense

class Model(object):
    def __init__(self, name, input_dim, n_class=None):
        self.name = name
        self._n_class = n_class
        self._input_dim = input_dim
        self.__z = {}
        self.__e = {}
        self.__pool_index = {}
        self.__layer_block_dct = {}
        self.__layer_name_lst = ['x']
        self.__layer_output_dim_lst = [input_dim]

        self.__train_x_set = None
        self.__train_y_set = None

        self.__test_x_set = None
        self.__test_y_set = None

        self.__train_loss_log = []
        self.__train_acc_log = []

        self.__test_loss_log = []
        self.__test_acc_log = []

    def initial(self, block):
        temp_dim = self._input_dim
        for i, layer_block in enumerate(block):
            name, temp_dim = layer_block.initial(temp_dim)
            if name not in self.__layer_name_lst:
                self.__layer_name_lst.append(name)
                self.__layer_output_dim_lst.append(temp_dim)
                self.__layer_block_dct[name] = layer_block
            else:
                print('Repeated Layer Name: {}!'.format(name))
                exit(1)
        self.print_structure()
    
    def save_model(self,path):
        file = open(path+'z.pickle', 'wb')
        pickle.dump(self.__z, file)
        file.close()
        file = open(path+'e.pickle', 'wb')
        pickle.dump(self.__e, file)
        file.close()
        file = open(path+'pool.pickle', 'wb')
        pickle.dump(self.__pool_index, file)
        file.close()
        file = open(path+'block.pickle', 'wb')
        pickle.dump(self.__layer_block_dct, file)
        file.close()
        file = open(path+'name.pickle', 'wb')
        pickle.dump(self.__layer_name_lst, file)
        file.close()
        file = open(path+'output.pickle', 'wb')
        pickle.dump(self.__layer_output_dim_lst, file)
        file.close()
    
    def load_model(self,path):
        with open(path+'z.pickle', 'rb') as file:
            self.__z =pickle.load(file)
        with open(path+'e.pickle', 'rb') as file:
            self.__e =pickle.load(file)
        with open(path+'pool.pickle', 'rb') as file:
            self.__e =pickle.load(file)
        with open(path+'block.pickle', 'rb') as file:
            self.__layer_block_dct =pickle.load(file)
        with open(path+'name.pickle', 'rb') as file:
            self.__layer_name_lst =pickle.load(file)
        with open(path+'output.pickle', 'rb') as file:
            self.__layer_output_dim_lst =pickle.load(file)


    def print_structure(self):
        for i in range(len(self.__layer_name_lst)):
            print("{}:Layer[{}] Output Dim={}".format(self.name,
                                                      self.__layer_name_lst[i], self.__layer_output_dim_lst[i]))
    def show_block(self,name):
       m=self.__layer_block_dct[name]
       m.show_filter()

    def __forward(self, _x_set):
        temp_z_set = _x_set.copy()
        self.__z['x'] = temp_z_set
        for layer_block in self.__layer_block_dct.values():
            if isinstance(layer_block, (Conv2D, Dense, Flatten, Activation)):
                temp_z_set = layer_block.forward(temp_z_set)
                self.__z[layer_block.name] = temp_z_set
            elif isinstance(layer_block, MaxPooling2D):
                temp_z_set, self.__pool_index[layer_block.name] = layer_block.forward(temp_z_set)
                self.__z[layer_block.name] = temp_z_set

    def __backward(self, _target_set):
        _y_set = self.__z[self.__layer_name_lst[-1]]
        self.__e[self.__layer_name_lst[-1]] = np.sum(-_target_set * np.log(_y_set + 1e-8))
        self.__e[self.__layer_name_lst[-2]] = self.__cross_entropy_cost(_y_set, _target_set)
        for i in range(len(self.__layer_name_lst) - 2, 0, -1):
            layer_name = self.__layer_name_lst[i]
            layer_name_down = self.__layer_name_lst[i - 1]
            layer_block = self.__layer_block_dct[layer_name]
            if isinstance(layer_block, (Conv2D, Dense, Flatten)):
                _e_set = self.__e[layer_name]
                self.__e[layer_name_down] = layer_block.backward(_e_set)
            elif isinstance(layer_block, MaxPooling2D):
                _e_set = self.__e[layer_name]
                self.__e[layer_name_down] = layer_block.backward(_e_set, self.__pool_index[layer_name])
            elif isinstance(layer_block, Activation):
                _e_set = self.__e[layer_name]
                self.__e[layer_name_down] = layer_block.backward(_e_set, self.__z[layer_name_down])

    @staticmethod
    def __cross_entropy_cost(_y_set, _target_set):
        prd_prb = _y_set.copy()
        if len(prd_prb) != len(_target_set):
            print("Cross entropy error!")
            exit(1)
        return prd_prb - _target_set

    def __gradient(self, _x_set, _target_set):
        # _dw = {}
        # _db = {}
        _g = {}
        self.__forward(_x_set)
        self.__backward(_target_set)
        _batch_train_loss = self.__loss_of_current() / len(_x_set)
        _batch_train_acc = 0
        for i in range(len(_x_set)):
            if np.argmax(self.__z[self.__layer_name_lst[-1]][i]) == np.argmax(_target_set[i]):
                _batch_train_acc += 1
        _batch_train_acc /= len(_x_set)
        for i in range(len(self.__layer_name_lst) - 1, 0, -1):
            layer_name = self.__layer_name_lst[i]
            layer_name_down = self.__layer_name_lst[i - 1]
            layer_block = self.__layer_block_dct[layer_name]
            if isinstance(layer_block, (Conv2D, Dense)):
                _z_down = self.__z[layer_name_down]
                _e = self.__e[layer_name]
                # _dw[layer_name], _db[layer_name] = layer_block.gradient(_z_down, _e)
                _g[layer_name] = layer_block.gradient(_z_down, _e)
        return _g, _batch_train_loss, _batch_train_acc

    def __gradient_descent(self, _g):
        for i in range(len(self.__layer_name_lst) - 1, 0, -1):
            layer_name = self.__layer_name_lst[i]
            layer_block = self.__layer_block_dct[layer_name]
            if isinstance(layer_block, (Conv2D, Dense)):
                layer_block.gradient_descent(_g[layer_name])

    def fit(self, train_x_set, train_y_set):
        self.__train_x_set = train_x_set
        self.__train_y_set = train_y_set
    def set_test(self, test_x_set, test_y_set):
        self.__test_x_set = test_x_set
        self.__test_y_set = test_y_set
    @staticmethod
    def __shuffle_set(sample_set, target_set):
        index = np.arange(len(sample_set))
        np.random.shuffle(index)
        return sample_set[index], target_set[index]

    def train(self, lr, momentum=0.9, max_epoch=1000, batch_size=64, shuffle=True, interval=1):
      
        #Training model by SGD optimizer.
        #:param lr: learning rate
        #:param momentum: momentum rate
        #:param max_epoch: max epoch
        #:param batch_size: batch size
        #:param shuffle: whether shuffle training set
        #:param interval: print training log each interval
        #:return: none

        if self.__train_x_set is None:
            print("None data fit!")
            exit(1)
        _vg = {}
        for layer_name, layer_block in zip(self.__layer_block_dct.keys(), self.__layer_block_dct.values()):
            if isinstance(layer_block, (Conv2D, Dense)):
                weight_shape = layer_block.weight_shape()
                _vg[layer_name] = \
                    {weight_name: np.zeros(list(weight_shape[weight_name])) for weight_name in weight_shape}
        batch_nums = len(self.__train_x_set) // batch_size
        for e in range(max_epoch + 1):
            if shuffle and e % batch_nums == 0:
                self.__shuffle_set(self.__train_x_set, self.__train_y_set)
            start_index = e % batch_nums * batch_size
            t_x = self.__train_x_set[start_index:start_index + batch_size]
            t_y = self.__train_y_set[start_index:start_index + batch_size]
            _g, _batch_train_loss, _batch_train_acc = self.__gradient(t_x, t_y)


            
            _g_test, _batch_test_loss, _batch_test_acc = self.__gradient(self.__test_x_set[:100,:], self.__test_y_set[:100,:])
            for layer_name, layer_block in zip(self.__layer_block_dct.keys(), self.__layer_block_dct.values()):
                if isinstance(layer_block, (Conv2D, Dense)):
                    for weight_name in _g[layer_name]:
                        _vg[layer_name][weight_name] = momentum * _vg[layer_name][weight_name] \
                                                       - lr * _g[layer_name][weight_name]
                        _g[layer_name][weight_name] = -_vg[layer_name][weight_name]
            self.__gradient_descent(_g)
            if interval and e % interval == 0:
                # print the training log of whole training set rather than batch:
                # train_acc = self.measure(self.__train_x_set, self.__train_y_set)
                self.__train_loss_log.append(_batch_train_loss)
                self.__train_acc_log.append(_batch_train_acc)
                self.__test_loss_log.append(_batch_test_loss)
                self.__test_acc_log.append(_batch_test_acc)
                print('Epoch[{}] Batch[{}] Batch_Train_Loss=[{}] Batch_Train_Acc=[{}]'
                      .format(e, e % batch_nums, _batch_train_loss, _batch_train_acc))
        
        plt.figure(1,figsize=(800, 800))
        plt.plot(self.__train_loss_log)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.figure(2,figsize=(800, 800))
        plt.plot(self.__train_acc_log)
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.figure(1,figsize=(800, 800))
        plt.plot(self.__test_loss_log)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.figure(2,figsize=(800, 800))
        plt.plot(self.__test_acc_log)
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.show()




    def predict(self, _x_set):
        self.__forward(_x_set)
        return np.argmax(self.__z[self.__layer_name_lst[-1]], axis=-1)

    def __loss_of_current(self):
        return self.__e[self.__layer_name_lst[-1]]

    def measure(self, _x_set, _target_set):
        _prd_set = self.predict(_x_set)
        _target_set = np.argmax(_target_set, axis=-1)
        _acc = 0
        for i in range(len(_x_set)):
            if _prd_set[i] == _target_set[i]:
                _acc += 1
        return _acc / len(_x_set)



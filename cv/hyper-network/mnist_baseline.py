import cPickle as pickle
import sys

import mxnet as mx
from mx_initializer import PReLUInitializer
from lr_scheduler import FactorScheduler
from mx_layers import *
from mx_solver import MXSolver
from data_utilities import load_mnist

network = variable('data')
network = convolution(X=network, n_filters=16, kernel_shape=(5, 5), stride=(1, 1), pad=(2, 2))
network = ReLU(network)
network = pooling(X=network, mode='maximum', kernel_shape=(2, 2), stride=(2, 2))
network = convolution(X=network, n_filters=16, kernel_shape=(5, 5), stride=(1, 1), pad=(2, 2))
network = ReLU(network)
network = pooling(X=network, mode='maximum', kernel_shape=(2, 2), stride=(2, 2))
network = flatten(network)
network = fully_connected(X=network, n_hidden_units=10)
network = softmax_loss(prediction=network, normalization='batch', id='softmax')

'''
original_data = load_mnist(shape=(1, 28, 28))
shrinked_data = load_mnist(path='shrinked_mnist', shape=(1, 28, 28))
data = original_data[:4] + shrinked_data[4:]
'''
data = load_mnist(shape=(1, 28, 28))

BATCH_SIZE = 256
lr = 0.001

optimizer_settings = {
  'initial_lr' : lr,
  'optimizer'  : 'Adam',
}

solver = MXSolver(
  batch_size = BATCH_SIZE,
  devices = (0, 1, 2, 3),
  epochs = 50,
  initializer = PReLUInitializer(),
  optimizer_settings = optimizer_settings,
  symbol = network,
  verbose = True,
)

info = solver.train(data)

identifier = 'mnist-baseline-network'
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))

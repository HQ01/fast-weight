import cPickle as pickle
from argparse import ArgumentParser

import mx_layers as layers
from mx_initializer import PReLUInitializer
from mx_solver import MXSolver
from data_utilities import load_mnist

parser = ArgumentParser()
parser.add_argument('--gpu_index', type=int)
configs = parser.parse_args()

def _normalized_convolution(**args):
  network = layers.convolution(**args)
  network = layers.batch_normalization(network)
  network = layers.ReLU(network)
  return network

network = layers.variable('data')
network = _normalized_convolution(X=network, n_filters=16, kernel_shape=(5, 5), stride=(1, 1), pad=(2, 2))
network = layers.pooling(X=network, mode='maximum', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))
# reduction of generalization performance
'''
network = _normalized_convolution(X=network, n_filters=16, kernel_shape=(5, 5), stride=(1, 1), pad=(2, 2))
network = layers.pooling(X=network, mode='maximum', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))
'''

history = []
for index in range(5):
  network = _normalized_convolution(X=network, n_filters=16, kernel_shape=(3, 3), stride=(1, 1), pad=(1, 1))
  history.append(network)
network = network + sum(history)
network = layers.pooling(X=network, mode='average', global_pool=True, kernel_shape=(28, 28), stride=(1, 1), pad=(0, 0))
network = layers.flatten(network)
network = layers.fully_connected(X=network, n_hidden_units=10)
network = layers.softmax_loss(prediction=network, normalization='batch', id='softmax')

optimizer_settings = {'args' : {'momentum' : 0.9}, 'initial_lr' : 0.1, 'optimizer'  : 'SGD'}

solver = MXSolver(
  batch_size         = 64,
  devices            = (configs.gpu_index,),
  epochs             = 50,
  initializer        = PReLUInitializer(),
  optimizer_settings = optimizer_settings,
  symbol             = network,
  verbose            = True,
)

training_data, training_labels, _, _, _, _ = load_mnist(shape=(1, 28, 28))
_, _, validation_data, validation_labels, test_data, test_labels = load_mnist(path='shrinked_mnist', shape=(1, 28, 28))
data = training_data, training_labels, validation_data, validation_labels, test_data, test_labels

info = solver.train(data)

identifier = 'shrinked-mnist-fixed-attention-network'
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))

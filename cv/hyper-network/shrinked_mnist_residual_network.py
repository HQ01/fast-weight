from argparse import ArgumentParser
import cPickle as pickle

import mx_layers as layers
from mx_initializer import PReLUInitializer
from mx_solver import MXSolver
from data_utilities import load_mnist

parser = ArgumentParser()
parser.add_argument('--gpu_index', type=int, required=True)
parser.add_argument('--n_residual_layers', type=int, required=True)
parser.add_argument('--postfix', type=str, required=True)
configs = parser.parse_args()

def _normalized_convolution(**args):
  network = layers.convolution(**args)
  network = layers.batch_normalization(network)
  network = layers.ReLU(network)
  return network

# TODO calculate receptive field
network = layers.variable('data')
for index in range(3):
  network = _normalized_convolution(X=network, n_filters=16, kernel_shape=(5, 5), stride=(1, 1), pad=(2, 2))
  network = layers.pooling(X=network, mode='maximum', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))

shared_weight = layers.variable('shared_convolution_weight')
shared_bias = layers.variable('shared_convolution_bias')
kwargs = {'n_filters' : 16, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)}
for index in range(configs.n_residual_layers):
  identity = network
  residual = _normalized_convolution(X=network, weight=shared_weight, bias=shared_bias, **kwargs)
  network = identity + residual

network = layers.pooling(X=network, mode='average', global_pool=True, kernel_shape=(1, 1), stride=(1, 1), pad=(0, 0))
network = layers.flatten(network)
network = layers.fully_connected(X=network, n_hidden_units=10)
network = layers.softmax_loss(prediction=network, normalization='batch', id='softmax')

optimizer_settings = {'args' : {'momentum' : 0.9}, 'initial_lr' : 0.1, 'optimizer'  : 'SGD'}

solver = MXSolver(
  batch_size         = 64,
  devices            = (configs.gpu_index,),
  epochs             = 30,
  initializer        = PReLUInitializer(),
  optimizer_settings = optimizer_settings,
  symbol             = network,
  verbose            = True,
)

data = []
data.extend(load_mnist(path='stretched_mnist', scale=1, shape=(1, 56, 56))[:2])
data.extend(load_mnist(path='stretched_canvas_mnist', scale=1, shape=(1, 56, 56))[2:])

info = solver.train(data)

identifier = 'shrinked-mnist-residual-network-%d-%s' % (configs.n_residual_layers, configs.postfix)
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))

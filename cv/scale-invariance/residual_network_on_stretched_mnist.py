import mx_layers as layers
from mx_initializer import PReLUInitializer
from mx_solver import MXSolver

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--gpu_index', type=int, default=0)
parser.add_argument('--n_residual_layers', type=int, required=True)
parser.add_argument('--postfix', type=str, default='')
args = parser.parse_args()

# TODO calculate receptive field

_convolution = lambda X : layers.convolution(X=X, n_filters=16, kernel_shape=(5, 5), stride=(1, 1), pad=(2, 2))

network = layers.variable('data')
for index in range(3):
  network = _convolution(network)
  network = layers.ReLU(network)
  network = layers.pooling(X=network, mode='maximum', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))

shared_weight = layers.variable('shared_weight')
shared_gamma = layers.variable('shared_gamma')
shared_beta = layers.variable('shared_beta')

_convolution = lambda X : layers.convolution(
  X=X, n_filters=16, kernel_shape=(3, 3), stride=(1, 1), pad=(1, 1), weight=shared_weight, no_bias=True
# X=X, n_filters=16, kernel_shape=(5, 5), stride=(1, 1), pad=(2, 2), weight=shared_weight, no_bias=True
)
for index in range(args.n_residual_layers):
  network = layers.batch_normalization(network, beta=shared_beta, gamma=shared_gamma, fix_gamma=False)
  network = layers.ReLU(network)
  network += _convolution(network)

network = layers.pooling(X=network, mode='average', kernel_shape=(7, 7), stride=(1, 1), pad=(0, 0))
# network = layers.pooling(X=network, mode='average', kernel_shape=(14, 14), stride=(1, 1), pad=(0, 0))
network = layers.flatten(network)
network = layers.fully_connected(X=network, n_hidden_units=10)
network = layers.softmax_loss(prediction=network, normalization='batch', id='softmax')

optimizer_settings = {'args' : {'momentum' : 0.9}, 'initial_lr' : 0.1, 'optimizer'  : 'SGD'}

solver = MXSolver(
  batch_size         = 64,
  devices            = (args.gpu_index,),
  epochs             = 30,
  initializer        = PReLUInitializer(),
  optimizer_settings = optimizer_settings,
  symbol             = network,
  verbose            = True,
)

from data_utilities import load_mnist
data = load_mnist(path='stretched_canvas_mnist', scale=1, shape=(1, 56, 56))[:2]
data += load_mnist(path='stretched_mnist', scale=1, shape=(1, 56, 56))[2:]

info = solver.train(data)

postfix = '-' + args.postfix if args.postfix else ''
identifier = 'residual-network-on-stretched-mnist-%d%s' % (args.n_residual_layers, postfix)

import cPickle as pickle
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))

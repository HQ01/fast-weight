import mx_layers as layers
from mx_initializer import PReLUInitializer
from mx_solver import MXSolver

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--gpu_index', type=int, required=True)
parser.add_argument('--n_residual_layers', type=int, required=True)
parser.add_argument('--postfix', type=str, required=True)
args = parser.parse_args()

_convolution = lambda network : \
  layers.convolution(X=network, n_filters=16, kernel_shape=(5, 5), stride=(1, 1), pad=(2, 2), no_bias=True)

network = layers.variable('data')

for index in range(args.n_residual_layers):
  network = layers.batch_normalization(network)
  network = layers.ReLU(network)
  network += _convolution(network)

network = layers.pooling(X=network, mode='average', kernel_shape=(56, 56), stride=(1, 1), pad=(0, 0))
network = layers.flatten(network)
network = layers.fully_connected(X=network, n_hidden_units=10)
network = layers.softmax_loss(prediction=network, normalization='batch', id='softmax')

optimizer_settings = {'args' : {'momentum' : 0.9}, 'initial_lr' : 0.1, 'optimizer'  : 'SGD'}

solver = MXSolver(
  batch_size         = 64,
# devices            = (args.gpu_index,),
  devices            = (0, 1, 2, 3,),
  epochs             = 30,
  initializer        = PReLUInitializer(),
  optimizer_settings = optimizer_settings,
  symbol             = network,
  verbose            = True,
)

from data_utilities import load_mnist
data = tuple()
data += load_mnist(path='stretched_mnist', scale=1, shape=(1, 56, 56))[:2]
data += load_mnist(path='stretched_canvas_mnist', scale=1, shape=(1, 56, 56))[2:]

info = solver.train(data)

postfix = '-' + args.postfix if args.postfix else ''
identifier = 'plain-residual-network-%d%s' % (args.n_residual_layers, postfix)

import cPickle as pickle
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))

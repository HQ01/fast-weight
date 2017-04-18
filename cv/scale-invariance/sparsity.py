from argparse import ArgumentParser
import cPickle as pickle

import mx_layers as layers
from mx_initializer import PReLUInitializer
from mx_solver import MXSolver

parser = ArgumentParser()
parser.add_argument('--gpu_index', type=int, required=True)
parser.add_argument('--n_residual_layers', type=int, required=True)
parser.add_argument('--postfix', type=str, required=True)
args = parser.parse_args()

def _normalized_convolution(**args):
  network = layers.convolution(no_bias=True, **args)
  network = layers.batch_normalization(network, fix_gamma=False)
  network = layers.ReLU(network)
  return network

# TODO calculate receptive field
network = layers.variable('data')
for index in range(3):
  network = _normalized_convolution(X=network, n_filters=16, kernel_shape=(5, 5), stride=(1, 1), pad=(2, 2))
  network = layers.pooling(X=network, mode='maximum', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))

shared_weight = layers.variable('shared_weight')
shared_gamma = layers.variable('shared_gamma')
shared_beta = layers.variable('shared_beta')
kwargs = {'n_filters' : 16, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)}

sparsity_loss = 0

for index in range(args.n_residual_layers):
  network = layers.batch_normalization(network, beta=shared_beta, gamma=shared_gamma, fix_gamma=False)
  network = layers.ReLU(network)
  network = layers.dropout(network, 0.5)
  identity = network
  residual = layers.convolution(X=network, weight=shared_weight, no_bias=True, **kwargs)
  network = identity + residual

network = layers.pooling(X=network, mode='average', global_pool=True, kernel_shape=(1, 1), stride=(1, 1), pad=(0, 0))
network = layers.flatten(network)
network = layers.fully_connected(X=network, n_hidden_units=10)
softmax_loss = layers.softmax_loss(prediction=network, normalization='batch', id='softmax')
alpha = 1.0
network = alpha * softmax_loss + (1 - alpha) * sparsity_loss

optimizer_settings = {'args' : {'momentum' : 0.9}, 'initial_lr' : 0.1, 'optimizer'  : 'SGD'}

solver = MXSolver(
  batch_size         = 64,
  devices            = (args.gpu_index,),
  epochs             = 100,
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
identifier = 'sparse-residual-network-on-shrinked-mnist-%d%s' % (args.n_residual_layers, postfix)
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))

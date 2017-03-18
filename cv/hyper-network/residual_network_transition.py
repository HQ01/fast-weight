import sys
import cPickle as pickle

from data_utilities import load_cifar10_record
from lr_scheduler import AtIterationScheduler
from mx_initializer import PReLUInitializer
import mx_layers as layers
from mx_solver import MXSolver

def _normalized_convolution(**args):
  network = layers.convolution(**args)
  network = layers.batch_normalization(network)
  network = layers.ReLU(network)
  return network

def _transit(network, n_filters, mode):
  left = _normalized_convolution(X=network, n_filters=n_filters / 2, kernel_shape=(3, 3), stride=(2, 2), pad=(1, 1))
  left = _normalized_convolution(X=left, n_filters=n_filters, kernel_shape=(3, 3), stride=(1, 1), pad=(1, 1))
  right = layers.pooling(X=network, mode='average', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))
  pad_width = (0, 0, 0, n_filters / 2, 0, 0, 0, 0)
  right = layers.pad(right, pad_width, 'constant')
  if mode is 'left': return left
  elif mode is 'right' : return right
  else: return left + right

N = int(sys.argv[1])
mode = sys.argv[2]

network = layers.variable('data')
network = layers.batch_normalization(network)

network = _normalized_convolution(X=network, n_filters=16, kernel_shape=(3, 3), stride=(1, 1), pad=(1, 1))
for index in range(N):
  identity = network
  residual = _normalized_convolution(X=network, n_filters=16, kernel_shape=(3, 3), stride=(1, 1), pad=(1, 1))
  residual = _normalized_convolution(X=residual, n_filters=16, kernel_shape=(3, 3), stride=(1, 1), pad=(1, 1))
  network = identity + residual

network = _transit(network, 32, mode)
for index in range(N):
  identity = network
  residual = _normalized_convolution(X=network, n_filters=32, kernel_shape=(3, 3), stride=(1, 1), pad=(1, 1))
  residual = _normalized_convolution(X=residual, n_filters=32, kernel_shape=(3, 3), stride=(1, 1), pad=(1, 1))
  network = identity + residual

network = _transit(network, 64, mode)
for index in range(N):
  identity = network
  residual = _normalized_convolution(X=network, n_filters=64, kernel_shape=(3, 3), stride=(1, 1), pad=(1, 1))
  residual = _normalized_convolution(X=residual, n_filters=64, kernel_shape=(3, 3), stride=(1, 1), pad=(1, 1))
  network = identity + residual

network = layers.pooling(X=network, mode='average', global_pool=True, kernel_shape=(8, 8), stride=(1, 1), pad=(1, 1))
network = layers.flatten(network)
network = layers.fully_connected(X=network, n_hidden_units=10)
network = layers.softmax_loss(prediction=network, normalization='batch', id='softmax')

BATCH_SIZE = 128
lr = 0.1
lr_table = {32000 : lr * 0.1, 48000 : lr * 0.01}
lr_scheduler = AtIterationScheduler(lr, lr_table)

optimizer_settings = {
  'args'         : {'momentum' : 0.9},
  'initial_lr'   : lr,
  'lr_scheduler' : lr_scheduler,
  'optimizer'    : 'SGD',
  'weight_decay' : 0.0001,
}

solver = MXSolver(
  batch_size = BATCH_SIZE,
  devices = (0, 1, 2, 3),
  epochs = 150,
  initializer = PReLUInitializer(),
  optimizer_settings = optimizer_settings,
  symbol = network,
  verbose = True,
)

data = load_cifar10_record(BATCH_SIZE)
info = solver.train(data)
identifier = 'cifar-residual-network-%d-%s-transition' % (N, mode)
pickle.dump(info, open('info/%s' % identifier, 'wb'))

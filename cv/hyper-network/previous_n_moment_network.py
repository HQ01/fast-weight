from argparse import ArgumentParser

import mx_layers as layers
from mx_utility import output_shape

from data_utilities import load_cifar10_record
from gpu_utility import gpu_availability
from lr_scheduler import AtIterationScheduler
from mx_initializer import PReLUInitializer
from mx_solver import MXSolver

def _normalized_convolution(**args):
  network = layers.convolution(**args)
  network = layers.batch_normalization(network)
  network = layers.ReLU(network)
  return network

def _transit(network, n_filters):
  left = _normalized_convolution(X=network, n_filters=n_filters / 2, kernel_shape=(3, 3), stride=(2, 2), pad=(1, 1))
  left = _normalized_convolution(X=left, n_filters=n_filters, kernel_shape=(3, 3), stride=(1, 1), pad=(1, 1))
  right = layers.pooling(X=network, mode='average', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))
  pad_width = (0, 0, 0, n_filters / 2, 0, 0, 0, 0)
  right = layers.pad(right, pad_width, 'constant')
  return left + right

def _previous_n_moment_module(network, settings):
  n_layers = settings['n_layers']
  n_moments = settings['n_moments'] # current output plus previous n moment outputs
  convolution_args = settings['convolution_args']
  memory = [network]
  for index in range(n_layers):
    network = _normalized_convolution(X=network, **convolution_args)
    network = _normalized_convolution(X=network, **convolution_args)
    memory.append(network)
    print len(memory[max(0, index - n_moments + 2):])
    network = sum(memory[max(0, index - n_moments + 2):])
  return network

parser = ArgumentParser()
parser.add_argument('--n_layers', type=int)
parser.add_argument('--n_moments', type=int)
parser.add_argument('--gpu', type=str)
configs = parser.parse_args()
settings = {
  'n_layers'         : configs.n_layers,
  'n_moments'        : configs.n_moments,
  'convolution_args' : {'n_filters' : 16, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)},
}

network = layers.variable('data')
network = _normalized_convolution(X=network, n_filters=16, kernel_shape=(3, 3), stride=(1, 1), pad=(1, 1))
network = _previous_n_moment_module(network, settings)

for n_filters in (32, 64):
  settings['convolution_args']['n_filters'] = n_filters
  network = _transit(network, n_filters)
  network =_previous_n_moment_module(network, settings)

network = layers.pooling(X=network, mode='average', global_pool=True, kernel_shape=(8, 8), stride=(1, 1), pad=(1, 1))
network = layers.flatten(network)
network = layers.fully_connected(X=network, n_hidden_units=10)
network = layers.softmax_loss(prediction=network, normalization='batch', id='softmax')

print output_shape(network, data=(1000, 3, 32, 32))

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

DEVICES = map(int, configs.gpu.split(','))
solver = MXSolver(
  batch_size = BATCH_SIZE,
  devices = DEVICES,
  epochs = 150,
  initializer = PReLUInitializer(),
  optimizer_settings = optimizer_settings,
  symbol = network,
  verbose = True,
)

'''
data = load_cifar10_record(BATCH_SIZE)
info = solver.train(data)

identifier = 'cifar-previous-n-moment-network-%d-decaying-rate-%f' % (configs.n_layers, configs.n_moments)
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
'''

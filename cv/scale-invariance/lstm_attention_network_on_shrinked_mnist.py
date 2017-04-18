import mx_layers as layers
from mx_initializer import PReLUInitializer
from mx_solver import MXSolver

def _normalized_convolution(**kwargs):
  network = layers.convolution(no_bias=True, **kwargs)
  network = layers.batch_normalization(network, fix_gamma=False)
  network = layers.ReLU(network)
  return network

def _lstm_convolution(X, n_filters, weight):
  return \
    layers.convolution(X=X, n_filters=n_filters, kernel_shape=(1, 1), stride=(1, 1), pad=(0, 0), weight=weight, no_bias=True)

def _lstm(X, settings, parameters, memory):
  n_filters = settings['n_filters'] * 4
  X_weight, h_weight, bias = parameters
  previous_h, previous_c = memory

  if previous_h is 0: array = _lstm_convolution(X, n_filters, X_weight)
  else: array = \
      _lstm_convolution(X, n_filters, X_weight) + _lstm_convolution(previous_h, n_filters, h_weight)
  array = layers.broadcast_plus(array, bias)

  group = layers.slice(X=array, axis=1, n_outputs=4)
  i = layers.sigmoid(group[0])
  f = layers.sigmoid(group[1])
  o = layers.sigmoid(group[2])
  g = layers.tanh(group[3])

  next_c = f * previous_c + i * g
  next_h = o * layers.tanh(next_c)
  memory = next_h, next_c

  return memory

def _read(settings, memory):
  h, c = memory
  return h

def _write(X, settings, parameters, memory):
  return _lstm(X, settings, parameters, memory)

def _lstm_attention_module(network, settings):
  prefix = 'lstm_attention_module'

  n_filters = settings['convolution_settings']['n_filters']
  memory_settings = {'n_filters' : n_filters}
  X_weight = layers.variable('%s_X_weight' % prefix, shape=(4 * n_filters, n_filters, 1, 1))
  h_weight = layers.variable('%s_h_weight' % prefix, shape=(4 * n_filters, n_filters, 1, 1))
  lstm_bias = layers.variable('%s_lstm_bias' % prefix, shape=(1, 4 * n_filters, 1, 1))
  lstm_parameters = (X_weight, h_weight, lstm_bias)
  memory = 0, 0

  kwargs = {key : value for key, value in settings['convolution_settings'].items()}
  if settings['weight_sharing']:
    kwargs['weight'] = layers.variable('%s_weight' % prefix)
    kwargs['bias'] = layers.variable('%s_bias' % prefix)

  for index in range(settings['n_layers']):
    memory = _write(network, memory_settings, lstm_parameters, memory)
    network = _read(memory_settings, memory)
    network = _normalized_convolution(X=network, **kwargs)

  return network

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--gpu_index', type=int, required=True)
parser.add_argument('--n_layers', type=int, required=True)
parser.add_argument('--postfix', type=str, required=True)
args = parser.parse_args()

network = layers.variable('data')
for index in range(3):
  network = _normalized_convolution(X=network, n_filters=16, kernel_shape=(5, 5), stride=(1, 1), pad=(2, 2))
  network = layers.pooling(X=network, mode='maximum', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))

convolution_settings = {'n_filters' : 16, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)}
settings = {'convolution_settings' : convolution_settings, 'n_layers' : args.n_layers, 'weight_sharing' : True}
network = _lstm_attention_module(network, settings)

network = layers.pooling(X=network, mode='average', global_pool=True, kernel_shape=(1, 1), stride=(1, 1), pad=(0, 0))
network = layers.flatten(network)
network = layers.fully_connected(X=network, n_hidden_units=10)
network = layers.softmax_loss(prediction=network, normalization='batch', id='softmax')

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
identifier = 'lstm-attention-network-on-shrinked-mnist-%d%s' % (args.n_layers, postfix)

import cPickle as pickle
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))

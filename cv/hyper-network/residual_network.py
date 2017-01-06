import mxnet as mx
from mx_layers import *
from hyper_network import *

def _normalized_convolution(X, kernel_shape, n_kernels, stride, pad, weight=None, bias=None):
  # TODO BN settings
  network = convolution(X, kernel_shape, n_kernels, stride, pad, weight=weight, bias=bias)
  network = batch_normalization(network, fix_gamma=False)
  network = ReLU(network)
  return network

_WIDTH, _HEIGHT = 3, 3

def _transit(X, n_kernels):
  P = _normalized_convolution(X, (_WIDTH, _HEIGHT), n_kernels, (2, 2), (1, 1))
  P = _normalized_convolution(P, (_WIDTH, _HEIGHT), n_kernels, (1, 1), (1, 1))
  Q = pooling(X, 'average', (2, 2), (2, 2))
  Q = convolution(Q, (1, 1), n_kernels)
  return P + Q

def _recur(X, n_kernels, weight=None, bias=None):
  residual = _normalized_convolution(X, (_WIDTH, _HEIGHT), n_kernels, (1, 1), (1, 1), weight=weight, bias=bias)
  residual = \
    _normalized_convolution(residual, (_WIDTH, _HEIGHT), n_kernels, (1, 1), (1, 1), weight=weight, bias=bias)
  return X + residual

def triple_state_residual_network(n, **kwargs):
  mode = kwargs['mode']
  N_z = 64
  d = N_z
  global _WIDTH, _HEIGHT

  module_index = 0
  def _generate_weight(filter_in, filter_out):
    if mode == 'normal': return None
    elif mode == 'weight-sharing':
      weight = variable('shared_convolution_weight%d' % module_index)
      module_index += 1
      return weight
    elif mode == 'hyper':
      if kwargs['embedding'] == 'feature_map':
        embedding = generated_convolution_embedding(network, (_WIDTH, _HEIGHT), kwargs['batch_size'])
      if kwargs['embedding'] == 'parameter':
        embedding = N_z
      return generated_convolution_weight(embedding, d, filter_in, filter_out, _WIDTH, _HEIGHT)

  network = variable('data')

  FILTER_IN, FILTER_OUT = 16, 16
  network = _normalized_convolution(network, (_WIDTH, _HEIGHT), FILTER_IN, (1, 1), (1, 1))
  weight = _generate_weight(FILTER_IN, FILTER_OUT)
  for i in range(n): network = _recur(network, FILTER_OUT, weight=weight)

  FILTER_IN, FILTER_OUT = 32, 32
  network = _transit(network, FILTER_IN)
  weight = _generate_weight(FILTER_IN, FILTER_OUT)
  for i in range(n - 1): network = _recur(network, FILTER_OUT, weight=weight)

  FILTER_IN, FILTER_OUT = 64, 64
  network = _transit(network, 64)
  weight = _generate_weight(FILTER_IN, FILTER_OUT)
  for i in range(n - 1): network = _recur(network, 64, weight=weight)

  network = pooling(network, 'average', (8, 8), (1, 1))
  network = flatten(network)
  network = fully_connected(network, 10)
  network = softmax_loss(network)
  return network

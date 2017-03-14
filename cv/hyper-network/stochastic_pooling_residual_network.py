from random import random
import mx_layers as layers
from residual_network_frame import residual_network

def _normalized_convolution(network, kernel_shape, n_filters, stride, pad):
  network = layers.convolution(X=network, kernel_shape=kernel_shape, n_filters=n_filters, stride=stride, pad=pad)
  network = layers.batch_normalization(network)
  network = layers.ReLU(network)
  return network

def _normalized_pooling(network, mode, kernel_shape, stride, pad):
  network = layers.pooling(X=network, mode=mode, kernel_shape=kernel_shape, stride=stride, pad=pad)
  network = layers.batch_normalization(network)
  return network

def _random_gate(p, shape):
  return layers.maximum(0, layers.sign(p - layers.uniform(shape=shape)))

def _transit(network, module_index):
  n_filters = {0 : 16, 1 : 32, 2 : 64}[module_index]
  if module_index == 0:
    network = _normalized_convolution(network, (3, 3), n_filters, (2, 2), (1, 1))
    return network
  else:
    P = _normalized_convolution(network, (3, 3), n_filters, (2, 2), (1, 1))
    P = _normalized_convolution(P, (3, 3), n_filters, (1, 1), (1, 1))
    Q = layers.pooling(X=network, mode='average', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))
    Q = layers.pad(Q, (0, 0) + (0, n_filters / 2) + (0, 0) * 2, 'constant')
    return P + Q

def _recur(network, module_index, settings):
  _, shape, _ = network.infer_shape(data=(1, 3, 32, 32))
  n_filters = {0 : 16, 1 : 32, 2 : 64}[module_index]
  n_layers = settings['n_layers']
  p = settings.get('p', False)
  if module_index == 0: n_layers += 1
  for time in range(n_layers):
    identity = network
    if p > 0:
      long_path = _normalized_convolution(network, (3, 3), n_filters, (1, 1), (1, 1))
      long_path = _normalized_convolution(long_path, (3, 3), n_filters, (1, 1), (1, 1))
      short_path = _normalized_pooling(network, settings['mode'], (3, 3), (1, 1), (1, 1))
      gate = _random_gate(p, (1, 1, 1, 1))
      residual = gate * long_path + (1 - gate) * short_path
    else:
      residual = _normalized_convolution(network, (3, 3), n_filters, (1, 1), (1, 1))
      residual = _normalized_convolution(residual, (3, 3), n_filters, (1, 1), (1, 1))
    network = identity + residual
  return network

def stochastic_pooling_residual_network(settings):
  recur = lambda network, module_index : _recur(network, module_index, settings)
  procedures = ((_transit, recur),) * 3
  network = residual_network(procedures)
  return network

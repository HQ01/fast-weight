import mxnet as mx
from mx_layers import *
from hyper_network import *

def _normalized_convolution(X, kernel_shape, n_filters, stride, pad, weight=None, bias=None, name=None):
  from mxnet.symbol import Convolution
  # TODO BN settings
  convolution_name = None if name is None else '%s_convolution' % name
  kwargs = {
    'cudnn_mode'   : 'limited_workspace',
    'kernel_shape' : kernel_shape,
    'name'         : convolution_name,
    'n_filters'    : n_filters,
    'pad'          : pad,
    'stride'       : stride,
    'X'            : X,
  }
  if weight is not None: kwargs['weight'] = weight
  if bias is not None: kwargs['bias'] = bias
  network = convolution(**kwargs)
  bn_name = None if name is None else '%s_bn' % name
  network = batch_normalization(network, fix_gamma=False, name=bn_name)
  network = ReLU(network)
  return network

_WIDTH, _HEIGHT = 3, 3

def _transit(X, n_filters, index):
  name = 'transition%d' % index
  P = _normalized_convolution(X, (_WIDTH, _HEIGHT), n_filters, (2, 2), (1, 1), name='%s_P0' % name)
  P = _normalized_convolution(P, (_WIDTH, _HEIGHT), n_filters, (1, 1), (1, 1), name='%s_P1' % name)
  Q = pooling(X=X, mode='average', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))
  pad_width = (0, 0, 0, n_filters / 2, 0, 0, 0, 0)
  Q = pad(Q, pad_width, 'constant')
  return P + Q

def _recur(X, n_filters, weight=None, bias=None):
  residual = _normalized_convolution(X, (_WIDTH, _HEIGHT), n_filters, (1, 1), (1, 1), weight=weight, bias=bias)
  residual = \
    _normalized_convolution(residual, (_WIDTH, _HEIGHT), n_filters, (1, 1), (1, 1), weight=weight, bias=bias)
  return X + residual

def triple_state_residual_network(n, **kwargs):
  mode = kwargs['mode']
  N_z = 64
  d = N_z
  global _WIDTH, _HEIGHT

  module_index = 0
  def _generate_parameters(filter_in, filter_out, module_index):
    if mode == 'normal': weight, bias = None, None
    elif mode == 'weight-sharing':
      weight = variable('shared_convolution%d_weight' % module_index)
      bias = variable('shared_convolution%d_bias' % module_index)
    elif mode == 'hyper':
      if kwargs['embedding'] == 'feature_map':
        embedding = generated_convolution_embedding(network, (_WIDTH, _HEIGHT), kwargs['batch_size'])
      if kwargs['embedding'] == 'parameter':
        embedding = N_z
      weight = generated_convolution_weight(embedding, d, filter_in, filter_out, _WIDTH, _HEIGHT)
      bias = variable('shared_convolution_bias%d' % module_index)
    return weight, bias

  network = variable('data')

  FILTER_IN, FILTER_OUT = 16, 16
  network = _normalized_convolution(network, (_WIDTH, _HEIGHT), FILTER_IN, (1, 1), (1, 1), name='transition0')
  weight, bias = _generate_parameters(FILTER_IN, FILTER_OUT, 0)
  for i in range(n): network = _recur(network, FILTER_OUT, weight=weight, bias=bias)

  FILTER_IN, FILTER_OUT = 32, 32
  network = _transit(network, FILTER_IN, 1)
  weight, bias = _generate_parameters(FILTER_IN, FILTER_OUT, 1)
  for i in range(n - 1): network = _recur(network, FILTER_OUT, weight=weight, bias=bias)

  FILTER_IN, FILTER_OUT = 64, 64
  network = _transit(network, 64, 2)
  weight, bias = _generate_parameters(FILTER_IN, FILTER_OUT, 2)
  for i in range(n - 1): network = _recur(network, 64, weight=weight, bias=bias)

  network = pooling(X=network, mode='average', kernel_shape=(8, 8), stride=(1, 1), pad=(0, 0))
  network = flatten(network)
  network = fully_connected(network, 10)
  network = softmax_loss(network, normalization='batch')
  return network

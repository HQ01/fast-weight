import mxnet as mx
from mx_layers import *
from batch_convolution import batch_convolution
from hyper_network import *

def _normalized_convolution(X, kernel_shape, n_filters, stride, pad, weight=None, bias=None, name=None, constant=False):
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
  if constant: kwargs['attribute'] = {'lr_mult' : '0.0'} 
  network = convolution(**kwargs)
  bn_name = None if name is None else '%s_bn' % name
  network = batch_normalization(network, fix_gamma=False, name=bn_name)
  network = ReLU(network)
  return network

def _normalized_batch_convolution(X, kernel_shape, n_filters, stride, pad, data_shape, weight=None, bias=None, name=None):
  network = batch_convolution(X, kernel_shape, n_filters, stride, pad, data_shape, weight=weight, bias=bias)
  network = batch_normalization(network, fix_gamma=False)
  network = ReLU(network)
  return network

_WIDTH, _HEIGHT = 3, 3

def _transit(X, n_filters, index, constant=False):
  name = 'transition%d' % index
  P = _normalized_convolution(X, (_WIDTH, _HEIGHT), n_filters, (2, 2), (1, 1), name='%s_P0' % name, constant=constant)
  P = _normalized_convolution(P, (_WIDTH, _HEIGHT), n_filters, (1, 1), (1, 1), name='%s_P1' % name, constant=constant)
  Q = pooling(X=X, mode='average', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))
  pad_width = (0, 0, 0, n_filters / 2, 0, 0, 0, 0)
  Q = pad(Q, pad_width, 'constant')
  return P + Q

def _recur(X, n_filters, weight=None, bias=None, **kwargs):
  mode = kwargs.get('mode', 'normal')
  if mode is 'normal':
    residual = _normalized_convolution(X, (_WIDTH, _HEIGHT), n_filters, (1, 1), (1, 1), weight=weight, bias=bias)
    residual = \
      _normalized_convolution(residual, (_WIDTH, _HEIGHT), n_filters, (1, 1), (1, 1), weight=weight, bias=bias)
  elif mode is 'batch':
    data_shape = kwargs['data_shape']
    residual = \
      _normalized_batch_convolution(X, (_WIDTH, _HEIGHT), n_filters, (1, 1), (1, 1), data_shape, weight=weight, bias=bias)
    residual = \
      _normalized_batch_convolution(residual, (_WIDTH, _HEIGHT), n_filters, (1, 1), (1, 1), data_shape, weight=weight, bias=bias)
  return X + residual

def triple_state_residual_network(n, **kwargs):
  mode = kwargs['mode']
  constant = kwargs.get('constant_transition', False)
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
        weight = convolution_weight_from_feature_maps(network, filter_in, filter_out, kwargs['data_shape'])
      if kwargs['embedding'] == 'parameter':
        embedding = N_z
        weight = convolution_weight_from_parameters(embedding, d, filter_in, filter_out, _WIDTH, _HEIGHT) # hyper-network paper
      bias = variable('shared_convolution%d_bias' % module_index) # TODO
    return weight, bias

  network = variable('data')

  FILTER_IN, FILTER_OUT = 16, 16
  network = \
    _normalized_convolution(network, (_WIDTH, _HEIGHT), FILTER_IN, (1, 1), (1, 1), name='transition0', constant=constant)
  weight, bias = _generate_parameters(FILTER_IN, FILTER_OUT, 0)
  for i in range(n):
    if mode is 'hyper' and kwargs['embedding'] is 'feature_map':
      network = _recur(network, FILTER_OUT, mode='batch', weight=weight, bias=bias, data_shape=kwargs['data_shape'])
    else: network = _recur(network, FILTER_OUT, mode='normal', weight=weight, bias=bias)

  FILTER_IN, FILTER_OUT = 32, 32
  network = _transit(network, FILTER_IN, 1, constant)
  weight, bias = _generate_parameters(FILTER_IN, FILTER_OUT, 1)
  for i in range(n - 1):
    if mode is 'hyper' and kwargs['embedding'] is 'feature_map':
      network = _recur(network, FILTER_OUT, mode='batch', weight=weight, bias=bias, data_shape=kwargs['data_shape'])
    else: network = _recur(network, FILTER_OUT, mode='normal', weight=weight, bias=bias)

  FILTER_IN, FILTER_OUT = 64, 64
  network = _transit(network, FILTER_IN, 2, constant)
  weight, bias = _generate_parameters(FILTER_IN, FILTER_OUT, 2)
  for i in range(n - 1):
    if mode is 'hyper' and kwargs['embedding'] is 'feature_map':
      network = _recur(network, FILTER_OUT, mode='batch', weight=weight, bias=bias, data_shape=kwargs['data_shape'])
    else: network = _recur(network, FILTER_OUT, mode='normal', weight=weight, bias=bias)

  network = pooling(X=network, mode='average', kernel_shape=(8, 8), stride=(1, 1), pad=(0, 0))
  network = flatten(network)
  linear_transition_kwargs = {'X' : network, 'n_hidden_units' : 10, 'name' : 'linear_transition'}
  if constant: linear_transition_kwargs['attribute'] = {'lr_mult' : '0.0'}
  network = fully_connected(**linear_transition_kwargs)
  network = softmax_loss(network, normalization='batch')
  return network

# TODO cleanup
def _normal_transition(X, n_filters, index):
  # exclude identity connection
  name = 'transition%d' % index
  network = _normalized_convolution(X, (_WIDTH, _HEIGHT), n_filters, (2, 2), (1, 1), name='%s_convolution0' % name)
  network = _normalized_convolution(network, (_WIDTH, _HEIGHT), n_filters, (1, 1), (1, 1), name='%s_convolution1' % name)
  return network

# TODO cleanup
def transitional_network():
  network = variable('data')
  network = _normalized_convolution(network, (_WIDTH, _HEIGHT), 16, (1, 1), (1, 1), name='transition0')
  network = _normal_transition(network, 32, 1)
  network = _normal_transition(network, 64, 2)
  network = pooling(X=network, mode='average', kernel_shape=(8, 8), stride=(1, 1), pad=(0, 0))
  network = flatten(network)
  network = fully_connected(X=network, n_hidden_units=10, name='linear_transition')
  network = softmax_loss(network, normalization='batch')
  return network

import mxnet as mx
from mx_layers import *
from batch_convolution import batch_convolution
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

def _normalized_batch_convolution(X, kernel_shape, n_filters, stride, pad, data_shape, weight=None, bias=None, name=None):
  network = batch_convolution(X, kernel_shape, n_filters, stride, pad, data_shape, weight=weight, bias=bias)
  network = batch_normalization(network, fix_gamma=False)
  network = ReLU(network)
  return network

def _normalized_pooling(network, **kwargs):
  network = pooling(X=network, **kwargs)
  network = batch_normalization(network, fix_gamma=False)
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

def _generate_parameters(t, settings, previous_weight, previous_bias, cache):
  mode = settings['mode']
  if 'normal' in mode: weight, bias = None, None
  elif 'weight-sharing' in mode:
    prefix = cache['prefix']
    if previous_weight is None: weight = variable('%sshared%d_weight' % (prefix, t))
    else: weight = previous_weight
    if previous_bias is None: bias = variable('%sshared%d_bias' % (prefix, t))
    else: bias = previous_bias
  elif 'hyper' in mode:
    if settings['embedding'] == 'feature_map':
      weight = convolution_weight_from_feature_maps(network, filter_in, filter_out, settings['data_shape'])
    if settings['embedding'] == 'parameter':
      embedding = settings['N_z']
      weight = convolution_weight_from_parameters(embedding, embedding, filter_in, filter_out, _WIDTH, _HEIGHT)
    bias = variable('%sshared%d_bias' % (prefix, t))
  elif 'hybrid' in mode: # weight = slow_weight + fast_weight
    if cache is None:
      shared_weight = variable('%sshared%d_weight' % (prefix, t))
      shared_bias = variable('%sshared%d_bias' % (prefix, t))
      cache = {'shared_weight' : shared_weight, 'shared_bias' : shared_bias}
      step_weight = variable('%sstep%d_weight' % (prefix, t))
      gamma = settings['gamma']
      weight = gamma * shared_weight + (1 - gamma) * step_weight
      bias = gamma * shared_bias + (1 - gamma) * step_bias
  return weight, bias, cache

_recurrent_module_count = 0
def _recur(network, times, settings):
  global _recurrent_module_count
  prefix = 'recurrent_module%d' % _recurrent_module_count
  mode = settings['mode']
  filter_in, filter_out = settings['filter_in'], settings['filter_out']

  weight, bias = None, None
  cache = {'prefix' : 'recurrent_module%d' % _recurrent_module_count}
  args = ((_WIDTH, _HEIGHT), filter_out, (1, 1), (1, 1))

  for t in range(times):
    if 'hyper' in mode and settings['embedding'] is 'feature_map': kwargs = {'data_shape' : settings['data_shape']}
    else: kwargs = {}
    if 'normal' in mode: function = _normalized_convolution
    elif 'weight-sharing' in mode: function = _normalized_convolution
    elif 'hyper' in mode and settings['embedding'] is 'feature_map':
      function = _normalized_batch_convolution
    weight, bias, cache = _generate_parameters(t, settings, weight, bias, cache)
    kwargs.update({'weight' : weight, 'bias' : bias})
    residual = function(network, *args, **kwargs)
    residual = function(residual, *args, **kwargs)
    network = network + residual
    if '-p' in mode:
      for t_pooling in range(settings['pooling_times']):
        kwargs = {'mode' : 'average', 'kernel_shape' : args[0], 'stride' : args[2], 'pad' : args[3]}
        residual = _normalized_pooling(network, **kwargs)
        residual = _normalized_pooling(residual, **kwargs)
        network = network + residual

  _recurrent_module_count += 1
  return network

def triple_state_residual_network(settings):
  times = settings['times']
  network = variable('data')

  settings['filter_in'], settings['filter_out'] = 16, 16
  global _WIDTH, _HEIGHT
  network = _normalized_convolution(network, (_WIDTH, _HEIGHT), settings['filter_in'], (1, 1), (1, 1), name='transition0')
  network = _recur(network, times, settings)

  settings['filter_in'], settings['filter_out'] = 32, 32
  network = _transit(network, settings['filter_in'], 1)
  network = _recur(network, times - 1, settings)
  
  settings['filter_in'], settings['filter_out'] = 64, 64
  network = _transit(network, settings['filter_in'], 2)
  network = _recur(network, times - 1, settings)

  network = pooling(X=network, mode='average', kernel_shape=(8, 8), stride=(1, 1), pad=(0, 0))
  network = flatten(network)
  network = fully_connected(X=network, n_hidden_units=10, name='linear_transition')
  network = softmax_loss(network, normalization='batch')
  return network

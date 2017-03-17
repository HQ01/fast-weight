import mx_layers as layers
from mx_utility import output_shape

def _normalized_convolution(**args):
  network = layers.convolution(**args)
  network = layers.batch_normalization(network)
  network = layers.ReLU(network)
  return network

def _rnn_convolution(X, n_filters, weight):
  return layers.convolution(X=X, n_filters=n_filters, kernel_shape=(1, 1), stride=(1, 1), pad=(0, 0), weight=weight, no_bias=True)

def _rnn(X, settings, parameters, memory):
  n_filters = settings['n_filters'] * 4
  X_weight, h_weight, bias = parameters
  previous_h, previous_c = memory
  # TODO normalization
  if previous_h is 0:
    array = _rnn_convolution(X, n_filters, X_weight)
    array = layers.broadcast_plus(array, bias)
  else:
    array = _rnn_convolution(X, n_filters, X_weight) + _rnn_convolution(previous_h, n_filters, h_weight)
    array = layers.broadcast_plus(array, bias)
  group = layers.slice(X=array, axis=1, n_outputs=4)
  i = layers.sigmoid(group[0])
  f = layers.sigmoid(group[1])
  o = layers.sigmoid(group[2])
  g = layers.sigmoid(group[3])
  next_c = f * previous_c + i * g
  next_h = o * layers.tanh(next_c)
  memory = next_h, next_c
  return memory

def _read(settings, memory):
  n_filters = settings['n_filters']
  h, c = memory
  # TODO kernel shape?
  return _normalized_convolution(X=h, n_filters=n_filters, kernel_shape=(3, 3), stride=(1, 1), pad=(1, 1))
# return _normalized_convolution(X=h, n_filters=n_filters, kernel_shape=(1, 1), stride=(1, 1), pad=(0, 0))

def _write(X, settings, parameters, memory):
  memory = _rnn(X, settings, parameters, memory)
  return memory

_n_rnn_attention_module = 0
def _rnn_attention_module(network, settings):
  global _n_rnn_attention_module
  prefix = 'rnn_attention_module%d' % _n_rnn_attention_module

  n_filters = settings['convolution_settings']['n_filters']
  memory_settings = {'n_filters' : n_filters}
  X_weight = layers.variable('%s_X_weight' % prefix, shape=(4 * n_filters, n_filters, 1, 1))
  h_weight = layers.variable('%s_h_weight' % prefix, shape=(4 * n_filters, n_filters, 1, 1))
  rnn_bias = layers.variable('%s_rnn_bias' % prefix, shape=(1, 4 * n_filters, 1, 1))
  rnn_parameters = (X_weight, h_weight, rnn_bias)
  memory = (0, 0)

  kwargs = {key : value for key, value in settings['convolution_settings'].items()}
  if settings['weight_sharing']:
    kwargs['weight'] = layers.variable('%s_weight' % prefix)
    kwargs['bias'] = layers.variable('%s_bias' % prefix)
  for index in range(settings['n_layers']):
    memory = _write(network, memory_settings, rnn_parameters, memory) # dynamic period of memory writing
    network = _read(memory_settings, memory)
    network = _normalized_convolution(X=network, **kwargs)
    network = _normalized_convolution(X=network, **kwargs)

  _n_rnn_attention_module += 1
  return network

def _transit(network, n_filters):
  left = _normalized_convolution(X=network, n_filters=n_filters, kernel_shape=(3, 3), stride=(2, 2), pad=(1, 1))
  left = _normalized_convolution(X=left, n_filters=n_filters, kernel_shape=(3, 3), stride=(2, 2), pad=(1, 1))
  right = layers.pooling(X=network, mode='average', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))
  pad_width = (0, 0, 0, n_filters / 2, 0, 0, 0, 0)
  right = layers.pad(right, pad_width, 'constant')
  return right + right

def rnn_attention_network(settings):
  network = layers.variable('data')
  network = layers.batch_normalization(network)
  for module_settings in settings:
    if module_settings['operator'] is 'rnn_attention_module':
      network = _rnn_attention_module(network, module_settings['settings'])
    elif module_settings['operator'] is 'transit':
      network = _transit(network, module_settings['n_filters'])
    else:
      args = module_settings.get('args', tuple())
      kwargs = {key : value for key, value in module_settings.get('kwargs', {}).items()}
      if args: args = (network,) + args
      else: kwargs['X'] = network
      network = getattr(layers, module_settings['operator'])(*args, **kwargs)
  network = layers.pooling(X=network, mode='average', global_pool=True, kernel_shape=(1, 1), stride=(1, 1), pad=(1, 1))
  network = layers.flatten(network)
  network = layers.fully_connected(X=network, n_hidden_units=10)
  network = layers.softmax_loss(prediction=network, normalization='batch', id='softmax')
  return network

if __name__ is '__main__':
  settings = (
    {
      'operator' : 'convolution',
      'kwargs' : {'n_filters' : 16, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)},
    },
    {
      'operator' : 'pooling',
      'kwargs' : {'mode' : 'maximum', 'kernel_shape' : (2, 2), 'stride' : (2, 2), 'pad' : (0, 0)},
    },
    {
      'operator' : 'rnn_attention_module',
      'settings' : {
        'convolution_settings' : {'n_filters' : 16, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)},
        'n_layers' : 3,
        'weight_sharing' : True,
      },
    }
  )
  network = rnn_attention_network(settings)
  print output_shape(network, data=(10000, 16, 32, 32))

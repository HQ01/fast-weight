import mx_layers as layers
from mx_utility import output_shape

def _normalized_convolution(**args):
  network = layers.convolution(**args)
  network = layers.batch_normalization(network)
  network = layers.ReLU(network)
  return network

def _read(memory, settings):
  return memory

def _write(X, memory, settings):
  decaying_rate = settings['decaying_rate']
  learning_rate = settings['learning_rate']
  memory = decaying_rate * memory + learning_rate * X
  memory = layers.batch_normalization(memory)
  return memory

_n_decayed_attention_module = 0
def _decayed_attention_module(network, settings):
  global _n_decayed_attention_module
  prefix = 'decayed_attention_module%d' % _n_decayed_attention_module

  memory = 0
  memory_settings = settings['memory_settings']

  kwargs = {key : value for key, value in settings['convolution_settings'].items()}
  if settings['weight_sharing']:
    kwargs['weight'] = layers.variable('%s_weight' % prefix)
    kwargs['bias'] = layers.variable('%s_bias' % prefix)
  for index in range(settings['n_layers']):
    memory = _write(network, memory, memory_settings) # dynamic period of memory writing
    network = _read(memory, memory_settings)
    network = _normalized_convolution(X=network, **kwargs)
    network = _normalized_convolution(X=network, **kwargs)

  _n_decayed_attention_module += 1
  return network

def _transit(network, n_filters):
  left = _normalized_convolution(X=network, n_filters=n_filters, kernel_shape=(3, 3), stride=(2, 2), pad=(1, 1))
  left = _normalized_convolution(X=left, n_filters=n_filters, kernel_shape=(3, 3), stride=(2, 2), pad=(1, 1))
  right = layers.pooling(X=network, mode='average', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))
  pad_width = (0, 0, 0, n_filters / 2, 0, 0, 0, 0)
  right = layers.pad(right, pad_width, 'constant')
  return right + right

def decayed_attention_network(settings):
  network = layers.variable('data')
  network = layers.batch_normalization(network)
  for module_settings in settings:
    if module_settings['operator'] is 'decayed_attention_module':
      network = _decayed_attention_module(network, module_settings['settings'])
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
      'operator' : 'decayed_attention_module',
      'settings' : {
        'convolution_settings' : {'n_filters' : 16, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)},
        'n_layers' : 3,
        'memory_settings' : {'decaying_rate' : 0.5, 'learning_rate' : 0.5,},
      },
    }
  )
  network = decayed_attention_network(settings)
  print output_shape(network, data=(10000, 16, 32, 32))

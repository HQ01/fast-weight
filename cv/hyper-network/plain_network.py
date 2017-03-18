import mx_layers as layers
from mx_utility import output_shape

def _normalized_convolution(**args):
  network = layers.convolution(**args)
  network = layers.batch_normalization(network)
  network = layers.ReLU(network)
  return network

def _transit(network, n_filters):
  left = _normalized_convolution(X=network, n_filters=n_filters, kernel_shape=(3, 3), stride=(1, 1), pad=(1, 1))
  left = _normalized_convolution(X=left, n_filters=n_filters, kernel_shape=(3, 3), stride=(2, 2), pad=(1, 1))
  right = layers.pooling(X=network, mode='average', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))
  pad_width = (0, 0, 0, n_filters / 2, 0, 0, 0, 0)
  right = layers.pad(right, pad_width, 'constant')
  # return left + right # TODO right + right
  return left

def plain_network(settings):
  network = layers.variable('data')
  network = layers.batch_normalization(network)
  for module_settings in settings:
    if module_settings['operator'] is 'transit':
      network = _transit(network, module_settings['n_filters'])
    elif module_settings['operator'] is 'convolution':
      kwargs = {key : value for key, value in module_settings.get('kwargs', {}).items()}
      network = _normalized_convolution(X=network, **kwargs)
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
  settings = ({
    'operator' : 'convolution', 
    'kwargs' : {'n_filters' : 16, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)},
  },) + ({
      'operator' : 'convolution', 
      'kwargs' : {'n_filters' : 16, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)},
  },) * 6 + ({
    'operator' : 'transit',
    'n_filters' : 32
  },) + ({
      'operator' : 'convolution', 
      'kwargs' : {'n_filters' : 32, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)},
  },) * 6 + ({
    'operator' : 'transit',
    'n_filters' : 64
  },) + ({
      'operator' : 'convolution', 
      'kwargs' : {'n_filters' : 64, 'kernel_shape' : (3, 3), 'stride' : (1, 1), 'pad' : (1, 1)},
  },)
  network = plain_network(settings)
  print output_shape(network, data=(10000, 16, 32, 32))

import mx_layers as layers
from mx_utility import output_shape

global _DEBUG
_DEBUG = True

def _normalized_convolution(network, kernel_shape, n_filters, stride, pad):
  network = layers.convolution(X=network, kernel_shape=kernel_shape, n_filters=n_filters, stride=stride, pad=pad)
  network = layers.batch_normalization(network)
  network = layers.ReLU(network)
  return network

def _dense_module(network, settings):
  compactify = lambda network : _normalized_convolution(network, (1, 1), 16, (1, 1), (0, 0))
  convolve = lambda network : _normalized_convolution(network, (3, 3), 16, (1, 1), (1, 1))
  cache = [network]
  for index in range(settings['N_LAYERS']):
    network = layers.concatenate(
      X        = settings['CONNECT'](cache),
      axis     = 1,
      n_inputs = len(cache)
    )
    if _DEBUG: print output_shape(network, data=(10, 3, 32, 32))
    network = compactify(network)
    network = convolve(network)
    cache.append(network)
  return network

def dense_network(settings, n_classes=10):
  network = layers.variable('data')
  network = _normalized_convolution(network, (3, 3), 16, (1, 1), (1, 1))
  for module_settings in settings:
    network = _dense_module(network, module_settings)
  network = layers.pooling(X=network, mode='average', kernel_shape=(1, 1), stride=(1, 1), pad=(0, 0), global_pool=True)
  network = layers.flatten(network)
  network = layers.fully_connected(X=network, n_hidden_units=n_classes)
  network = layers.softmax_loss(network, normalization='batch')
  return network

if __name__ is '__main__':
  settings = ({'N_LAYERS' : 8, 'CONNECT' : lambda cache : cache},) * 1
  symbol = dense_network(settings)

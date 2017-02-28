import mx_layers as layers
from mx_utility import output_shape

def _normalized_convolution(network, kernel_shape, n_filters, stride, pad):
  network = layers.convolution(X=network, kernel_shape=kernel_shape, n_filters=n_filters, stride=stride, pad=pad)
  network = layers.batch_normalization(network)
  network = layers.ReLU(network)
  return network

def average(cache):
  return sum(cache) / len(cache)
def latest(cache):
  return cache[-1]

def unattentioned_network(times, function=average, n_classes=10):
  # TODO simplify network structure
  network = layers.variable('data')
  cache = []
  for time in range(times):
    network = _normalized_convolution(network, (3, 3), 16, (1, 1), (1, 1))
    cache.append(network)
  network = layers.batch_normalization(function(cache))
  network = _normalized_convolution(network, (3, 3), 16, (2, 2), (1, 1))
  network = _normalized_convolution(network, (3, 3), 16, (2, 2), (1, 1))
  network = layers.pooling(X=network, mode='average', kernel_shape=(8, 8), stride=(1, 1), pad=(0, 0))
  network = layers.fully_connected(X=network, n_hidden_units=n_classes)
  network = layers.softmax_loss(network, normalization='batch')
  return network

if __name__ is '__main__':
  network = unattentioned_network(4)

import mx_layers as layers
from mx_utility import output_shape

def _normalized_convolution(**args):
  network = layers.convolution(**args)
  network = layers.batch_normalization(network)
  network = layers.ReLU(network)
  return network

def naive_network(n_layers, weight_sharing):
  network = layers.variable('data')
  network = _normalized_convolution(X=network, n_filters=8, kernel_shape=(5, 5), stride=(1, 1), pad=(2, 2))
  network = layers.pooling(X=network, mode='maximum', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))
  if weight_sharing:
    shared_weight = layers.variable('shared_weight')
    shared_bias = layers.variable('shared_bias')
  for index in range(n_layers):
    if weight_sharing:
      network = _normalized_convolution(
        X            = network,
        n_filters    = 8,
        kernel_shape = (3, 3),
        stride       = (1, 1),
        pad          = (1, 1),
        weight       = shared_weight,
        bias         = shared_bias
      )
    else: network = _normalized_convolution(X=network, n_filters=16, kernel_shape=(3, 3), stride=(1, 1), pad=(1, 1))
  network = layers.pooling(X=network, mode='average', global_pool=True, kernel_shape=(1, 1), stride=(1, 1), pad=(1, 1))
  network = layers.flatten(network)
  network = layers.fully_connected(X=network, n_hidden_units=10)
  network = layers.softmax_loss(prediction=network, normalization='batch', id='softmax')
  return network

if __name__ is '__main__':
  network = naive_network(5, False)
  print output_shape(network, data=(10000, 3, 32, 32))
  for arg in network.list_arguments():
    if 'weight' in arg: print arg

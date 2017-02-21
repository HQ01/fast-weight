import mx_layers as layers
from residual_network_frame import residual_network
from unweighted_convolution import unweighted_convolution

def _normalized_weighted_convolution(network, kernel_shape, n_filters, stride, pad):
  network = layers.convolution(X=network, kernel_shape=kernel_shape, n_filters=n_filters, stride=stride, pad=pad)
  network = layers.batch_normalization(network)
  network = layers.ReLU(network)
  return network

def _normalized_unweighted_convolution(network, kernel_shape, n_filters, stride, pad):
  network = unweighted_convolution(
    X            = network,
    kernel_shape = kernel_shape,
    n_filters    = n_filters,
    stride       = stride,
    pad          = pad,
    data_shape   = (1, 3, 32, 32),
  )
  network = layers.batch_normalization(network)
  network = layers.ReLU(network)
  return network

def _transit(network, module_index):
  n_filters = {0 : 16, 1 : 32, 2 : 64}[module_index]
  if module_index == 0:
    network = _normalized_weighted_convolution(network, (3, 3), n_filters, (2, 2), (1, 1))
    return network
  else:
    P = _normalized_weighted_convolution(network, (3, 3), n_filters, (2, 2), (1, 1))
    P = _normalized_weighted_convolution(P, (3, 3), n_filters, (1, 1), (1, 1))
    Q = layers.pooling(X=network, mode='average', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))
    Q = layers.pad(Q, (0, 0) + (0, n_filters / 2) + (0, 0) * 2, 'constant')
    return P + Q

def _recur(network, module_index, settings):
  _, shape, _ = network.infer_shape(data=(10000, 3, 32, 32))
  n_filters = {0 : 16, 1 : 32, 2 : 64}[module_index]
  refining_times = settings['refining_times']
  if module_index == 0: refining_times += 1
  for time in range(refining_times):
    residual = _normalized_unweighted_convolution(network, (3, 3), n_filters, (1, 1), (1, 1))
    residual = _normalized_weighted_convolution(residual, (3, 3), n_filters, (1, 1), (1, 1))
    network = network + residual
  return network

def pooled_residual_network(refining_times):
  settings = {'refining_times' : refining_times}
  recur = lambda network, module_index : _recur(network, module_index, settings)
  procedures = ((_transit, recur),) * 3
  network = residual_network(procedures)
  return network

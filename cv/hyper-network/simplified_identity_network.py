import mx_layers as layers

def _normalized_convolution(network, kernel_shape, n_filters, stride, pad):
  network = layers.convolution(X=network, kernel_shape=kernel_shape, n_filters=n_filters, stride=stride, pad=pad)
  network = layers.batch_normalization(network)
  network = layers.ReLU(network)
  return network

def simplifed_identity_network(N):
  network = variable('data')
  for index in range(N):
    residual = _normalized_convolution(network, (3, 3), 16, (1, 1), (1, 1))
    residual = _normalized_convolution(network, (3, 3), 16, (1, 1), (1, 1))
    identity = network
    network = identity + residual
  network = layers.pooling(X=network, mode='average', global_pool=True, kernel_shape=(1, 1), stride=(1, 1), pad=(1, 1))
  network = layers.flatten(network)
  network = layers.fully_connected(X=network, n_hidden_units=10, name='linear_transition')
  network = layers.softmax_loss(prediction=network, normalization='batch', id='softmax')
  return network

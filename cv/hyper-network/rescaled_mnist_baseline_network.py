import mx_layers as layers

def _normalized_convolution(**args):
  network = layers.convolution(**args)
  network = layers.batch_normalization(network)
  network = layers.ReLU(network)
  return network

def naive_network(n_layers):
  network = layers.variable('data')
  shared_weight = layers.variable('shared_weight')
  shared_bias = layers.variable('shared_bias')
  for index in range(n_layers):
    network = _normalized_convolution(
      X            = network,
      n_filters    = 16,
      kernel_shape = (3, 3),
      stride       = (1, 1),
      pad          = (1, 1),
      weight       = shared_weight,
      bias         = shared_bias
    )
  network = layers.pooling(X=network, mode='average', global_pool=True, kernel_shape=(1, 1), stride=(1, 1), pad=(1, 1))
  network = layers.flatten(network)
  network = layers.fully_connected(X=network, n_hidden_units=10, name='linear_transition')
  network = layers.softmax_loss(prediction=network, normalization='batch', id='softmax')
  return network

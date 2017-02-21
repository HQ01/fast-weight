import mx_layers as layers

def residual_network(procedures):
  network = layers.variable('data')
  for index, procedure in enumerate(procedures):
    transit, recur = procedure
    network = transit(network, index)
    network = recur(network, index)
  network = layers.pooling(X=network, mode='average', kernel_shape=(1, 1), stride=(1, 1), pad=(1, 1), global_pool=True)
  network = layers.flatten(network)
  network = layers.fully_connected(X=network, n_hidden_units=10, name='linear_transition')
  network = layers.softmax_loss(network, normalization='batch')
  return network

import mx_layers as layers

def _normalized_convolution(network, kernel_shape, n_filters, stride, pad, weight=None, bias=None):
  args = {'X' : network, 'kernel_shape' : kernel_shape, 'n_filters' : n_filters, 'stride' : stride, 'pad' : pad}
  if weight is not None: args['weight'] = weight
  if bias is not None: args['bias'] = bias
  network = layers.convolution(**args)
  network = layers.batch_normalization(network)
  network = layers.ReLU(network)
  return network

def dual_activation_network(n_layers):
  shared_weight = layers.variable('shared_weight')
  shared_bias = layers.variable('shared_bias')
  network = layers.variable('data')
  network = _normalized_convolution(network, (3, 3), 16, (1, 1), (1, 1))
  for i in range(n_layers):
    private = _normalized_convolution(network, (3, 3), 16, (1, 1), (1, 1))
    shared = _normalized_convolution(network, (3, 3), 16, (1, 1), (1, 1), weight=shared_weight, bias=shared_bias)
    network =  private + shared
  network = layers.pooling(X=network, mode='average', global_pool=True, kernel_shape=(1, 1), stride=(1, 1), pad=(1, 1))
  network = layers.flatten(network)
  network = layers.fully_connected(X=network, n_hidden_units=10, name='linear_transition')
  network = layers.softmax_loss(prediction=network, normalization='batch', id='softmax')
  return network

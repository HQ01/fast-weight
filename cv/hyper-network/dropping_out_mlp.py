import mx_layers as layers

def _fully_connected(network, n_hidden_units, p):
  network = layers.fully_connected(X=network, n_hidden_units=n_hidden_units)
  network = layers.ReLU(network)
  network = layers.dropout(network, p)
  return network

def dropping_out_mlp(settings):
  network = layers.variable('data')
  network = layers.flatten(network)
  layer_settings = settings['layer_settings']
  for index, layer_setting in enumerate(layer_settings):
    n_hidden_units = layer_setting['n_hidden_units']
    p = layer_setting['p']
    network = _fully_connected(network, n_hidden_units, p)
  network = layers.fully_connected(X=network, n_hidden_units=settings['n_classes'])
  network = layers.softmax_loss(prediction=network, normalization='batch', id='softmax')
  return network

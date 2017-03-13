import mx_layers as layers
from mx_utility import output_shape

def _random_gate(p, shape):
  return layers.maximum(0, layers.sign(p - layers.uniform(shape=shape)))

def _fully_connected(network, n_hidden_units, mode, p):
  long_path = layers.fully_connected(X=network, n_hidden_units=n_hidden_units)
  long_path = layers.ReLU(long_path)
  short_path = layers.mean(network, axis=1)
  short_path = layers.reshape(short_path, (0, 1))
  short_path = layers.broadcast(short_path, (0, n_hidden_units))
  short_path = layers.ReLU(short_path)
  gate = _random_gate(p, (1, n_hidden_units))
  network = layers.broadcast_multiply(gate, long_path) + layers.broadcast_multiply(1 - gate, short_path)
  return network

def element_wise_stochastic_pooling_mlp(settings):
  network = layers.variable('data')
  network = layers.flatten(network)
  layer_settings = settings['layer_settings']
  for index, layer_setting in enumerate(layer_settings):
    n_hidden_units = layer_setting['n_hidden_units']
    mode = layer_setting['pooling_mode']
    p = layer_setting['p'] # the probability of using long path
    network = _fully_connected(network, n_hidden_units, mode, p)
  network = layers.fully_connected(X=network, n_hidden_units=settings['n_classes'])
  network = layers.softmax_loss(prediction=network, normalization='batch', id='softmax')
  return network

if __name__ is '__main__':
  settings = {
    'layer_settings' : (
      {'n_hidden_units' : 1024, 'pooling_mode' : 'average', 'p' : 0.5},
    ) * 3,
    'n_classes' : 10,
  }
  network = element_wise_stochastic_pooling_mlp(settings)
# print output_shape(network, data=(10, 3072))

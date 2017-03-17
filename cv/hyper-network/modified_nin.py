import mx_layers as layers

def _random_gate(p, shape):
  return layers.maximum(0, layers.sign(p - layers.uniform(shape=shape)))

def _activated_convolution(**kwargs):
  network = layers.convolution(**kwargs)
  return layers.ReLU(X=network)

def _transit(network, mode):
  if mode is 'convolution + dropout':
    network = _activated_convolution(X=network, kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))
    network = layers.dropout(network, 0.5)
  elif mode is 'pooling + dropout':
    network = layers.pooling(X=network, mode='average', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))
    network = layers.dropout(network, 0.5)
  elif mode is 'stochastic_pooling':
    network = _activated_convolution(X=network, kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))
    network = layers.pooling(X=network, mode='average', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))
  return network

def nin(settings):
  network = layers.variable('data')
  network = _activated_convolution(X=network, kernel_shape=(3, 3), n_filters=192, stride=(1, 1), pad=(1, 1))
  network = _activated_convolution(X=network, kernel_shape=(1, 1), n_filters=160, stride=(1, 1), pad=(0, 0))
  network = _activated_convolution(X=network, kernel_shape=(1, 1), n_filters=96, stride=(1, 1), pad=(0, 0))
  network = _transit(network, settings['transition_mode'])
  network = _activated_convolution(X=network, kernel_shape=(3, 3), n_filters=192, stride=(1, 1), pad=(1, 1))
  network = _activated_convolution(X=network, kernel_shape=(1, 1), n_filters=192, stride=(1, 1), pad=(0, 0))
  network = _activated_convolution(X=network, kernel_shape=(1, 1), n_filters=192, stride=(1, 1), pad=(0, 0))
  network = _transit(network, settings['transition_mode'])
  network = _activated_convolution(X=network, kernel_shape=(3, 3), n_filters=192, stride=(1, 1), pad=(1, 1))
  network = _activated_convolution(X=network, kernel_shape=(1, 1), n_filters=192, stride=(1, 1), pad=(0, 0))
  network = _activated_convolution(X=network, kernel_shape=(1, 1), n_filters=10, stride=(1, 1), pad=(0, 0))
  network = layers.pooling(X=network, mode='average', global_pool=True, kernel_shape=(1, 1), stride=(1, 1), pad=(0, 0))
  network = layers.flatten(network)
  network = layers.softmax_loss(prediction=network, normalization='batch', id='softmax')
  return network

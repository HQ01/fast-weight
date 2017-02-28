import mx_layers as layers
from mx_utility import output_shape
from batch_convolution import batch_convolution
from facility import mark

global _DEBUG
_DEBUG = True

def _normalized_convolution(network, kernel_shape, n_filters, stride, pad):
  args = {
    'X'            : network,
    'kernel_shape' : kernel_shape,
    'n_filters'    : n_filters,
    'stride'       : stride,
    'pad'          : pad
  }
  network = layers.convolution(**args)
  network = layers.batch_normalization(network)
  network = layers.ReLU(network)
  return network

def _normalized_batch_convolution(*args):
  network = batch_convolution(*args)
  network = layers.batch_normalization(network)
  network = layers.ReLU(network)
  return network

def _extract_representations(network, parameters, batch_size):
  variables = {'layer_index' : 0}
  def convolve(*args):
    layer_index = variables['layer_index']
    weight = parameters[layer_index]['weight']
    if weight is None: network = _normalized_convolution(*args)
    else:
      network = _normalized_batch_convolution(*(args + ((batch_size, 3, 32, 32), weight)))
    variables['layer_index'] += 1
    return network
  network = convolve(network, (3, 3), 16, (1, 1), (1, 1))
  network = layers.pooling(X=network, mode='maximum', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))
  network = convolve(network, (3, 3), 16, (1, 1), (1, 1))
  network = layers.pooling(X=network, mode='maximum', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))
  network = convolve(network, (3, 3), 16, (1, 1), (1, 1))
  network = layers.pooling(X=network, mode='maximum', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))
  return network

def _generate_parameters(network, kernel_shapes):
  weights = []
  for shape in kernel_shapes:
    width, height, depth = shape
    weight = layers.pooling(X=network, mode='average', kernel_shape=(2, 2), stride=(2, 2), pad=(1, 1))
    weight = _normalized_convolution(weight, (1, 1), depth, (1, 1), (0, 0))
#   weight = _normalized_convolution(network, (3, 3), depth, (1, 1), (1, 1))
#   weight = layers.pooling(X=weight, mode='average', kernel_shape=(2, 2), stride=(2, 2), pad=(1, 1))
    weights.append(weight)
  return weights

def recurrent_hypernetwork(T, batch_size):
  X = layers.variable('data')
  label = layers.variable('softmax_label')
  loss = 0
  parameters = ({'weight' : None, 'bias' : None}, {'weight' : None, 'bias' : None}, {'weight' : None, 'bias' : None})
  KERNEL_SHAPES = ((3, 3, 3 * 16),) + ((3, 3, 16 * 16),) * 2
  for time in range(T):
    network = _extract_representations(X, parameters, batch_size)
    prediction = layers.pooling(X=network, mode='average', global_pool=True, kernel_shape=(1, 1), stride=(1, 1), pad=(0, 0))
    prediction = layers.flatten(prediction)
    prediction = layers.fully_connected(X=prediction, n_hidden_units=10)
    loss += layers.softmax_loss(prediction=prediction, label=label)
    for index, weight in enumerate(_generate_parameters(network, KERNEL_SHAPES)):
      parameters[index]['weight'] = weight
  return loss

if __name__ is '__main__':
  print output_shape(recurrent_hypernetwork(3, 64), data=(64, 3, 32, 32))

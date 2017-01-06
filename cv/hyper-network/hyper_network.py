import mxnet as mx
from mx_layers import *

def generated_convolution_embedding(X, kernel, batch_size):
  # method 0
  X = pooling(X, 'average', kernel, (1, 1))
  X = flatten(X)
  X = mx.symbol.sum(X, axis=0) / float(batch_size)
  return X

def generated_convolution_weight(*args):
  # implementing the method proposed in "HyperNetworks"
  return _generated_convolution_weight._(*args)

class _generated_convolution_weight:
  _n = 0
  @staticmethod
  def _(embedding, d, filter_in, filter_out, width, height):
    # TODO layer name
    n = _generated_convolution_weight._n
    if isinstance(embedding, int):
      weight = variable('convolution_embedding%d_weight' % n, shape=(embedding,))
    else: weight = embedding
    _generated_convolution_weight._n += 1
    weight = fully_connected(weight, filter_in * d)
    weight = reshape(weight, (filter_in, d))
    weight = fully_connected(weight, filter_out * width * height)
    weight = reshape(weight, (filter_in, filter_out, width, height))
    weight = swap_axes(weight, 0, 1)
    return weight

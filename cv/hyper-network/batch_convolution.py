import mxnet as mx
from mx_layers import *
from mx_utility import output_shape
from facility import mark

_batch_convolution_count = 0
def batch_convolution(X, kernel_shape, n_filters, stride, pad, data_shape, weight=None, bias=None, **kwargs):
  global _batch_convolution_count
  prefix = 'batch_convolution%d' % _batch_convolution_count
  N, C, H, W = output_shape(X, data=data_shape)
  filter_in, filter_out = C, n_filters

  weight = reshape(weight, (-1, filter_in) + kernel_shape)
  weights = slice_channels(X=weight, n_outputs=N, axis=0)
  bias = variable('%s_bias' % prefix) if bias is None else bias

  networks = slice_channels(X=X, n_outputs=N, axis=0)
  networks = tuple(
    convolution(
      X=networks[i], kernel_shape=kernel_shape, n_filters=n_filters, stride=stride, pad=pad, weight=weights[i], bias=bias
    ) for i in range(N)
  )
  network = concatenate(X=networks, n_inputs=N, axis=0)

  _batch_convolution_count += 1
  return network

if __name__ == '__main__':
  X_SHAPE = (64, 3, 32, 32)
  network = variable('data')
  weight = variable('weight', shape=(64, 48, 3, 3))
  network = \
    batch_convolution(X=network, kernel_shape=(3, 3), n_filters=16, stride=(1, 1), pad=(1, 1), weight=weight, data_shape=X_SHAPE)
  print output_shape(network, data=X_SHAPE, weight=(64, 48, 3, 3))

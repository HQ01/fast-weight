import mxnet as mx
from mx_layers import *
import numpy as np0

from facility import mark

_batch_convolution_count = 0
def batch_convolution(X, kernel_shape, n_filters, stride, pad, data_shape, weight, **kwargs):
  global _batch_convolution_count
  _, X_shape, _ = X.infer_shape(data=data_shape)
  N, C, H, W = X_shape[0]
  filter_in, filter_out = C, n_filters

  weight = reshape(weight, (-1, filter_out) + kernel_shape)
  weights = slice_channels(X=weight, n_outputs=N, axis=0)
  bias = kwargs.get('bias', variable('batch_convolution%d_bias' % _batch_convolution_count))

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
  network = variable('data')
  weight = convolution(X=network, kernel_shape=(3, 3), n_filters=9, pad=(1, 1), stride=(1, 1))
  weight = pooling(X=weight, mode='maximum', kernel_shape=(8, 8), stride=(8, 8))
  network = \
    batch_convolution(X=network, kernel_shape=(4, 4), n_filters=3, stride=(1, 1), pad=(2, 2), weight=weight, data_shape=X_SHAPE)
  _, shape, _ = network.infer_shape(data=X_SHAPE)
  print shape

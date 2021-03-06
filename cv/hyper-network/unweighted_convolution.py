import mxnet as mx
from mxnet.symbol import broadcast_plus
from mx_layers import *
import numpy as np

from facility import mark

_n_unweighted_convolutions = 0
def unweighted_convolution(X, n_filters, kernel_shape, stride, pad, data_shape):
  _, X_SHAPES, _ = X.infer_shape(data=data_shape)
  X_SHAPE = X_SHAPES[0]
  N, C, H, W = X_SHAPE 
  spatial_kernel_shape = (C,) + kernel_shape
  spatial_pad = (0, 1, 1)
  spatial_stride = (1,) + stride
  global _n_unweighted_convolutions
  prefix = 'unweighted_convolution%d' % _n_unweighted_convolutions
  _n_unweighted_convolutions += 1
  network = reshape(X, (0, 1, C, H, W))
  pooled = pooling(
    X=network,
    mode='average',
    kernel_shape=spatial_kernel_shape,
    pad=spatial_pad,
    stride=spatial_stride,
  )
  shapes, _, _ = pooled.infer_shape(data=data_shape)
  pooled = reshape(pooled, (0, 1, H, W))
  shapes, _, _ = pooled.infer_shape(data=data_shape)
  bias = variable('%s_bias' % prefix, shape=(1, n_filters, 1, 1))
  network = broadcast_plus(pooled, bias)
  return network

if __name__ is '__main__':
  N, C = 1000, 3
  DATA_SHAPE = (N, C, 32, 32)
  data = np.random.normal(0, 1, DATA_SHAPE)
  N_FILTERS = 16
  BIAS_SHAPE = (1, N_FILTERS, 1, 1)
  bias = np.zeros(BIAS_SHAPE)
  network = variable('data')
  network = unweighted_convolution(network, N_FILTERS, (3, 3), (1, 1), (1, 1), DATA_SHAPE)
  args = {'data' : mx.nd.array(data, mx.gpu(0)), 'unweighted_convolution0_bias' : mx.nd.array(bias, mx.gpu(0))}
  executor = network.bind(mx.gpu(0), args) 
  executor.forward()
  result = executor.outputs[0].asnumpy()

import mxnet as mx
from mx_layers import *

def convolution_weight_from_feature_maps(X, filter_in, filter_out, data_shape):
  # dynamic filter network
  _, X_shape, _ = X.infer_shape(data=data_shape)
  N, C, H, W = X_shape[0]
  # method 0: reduce dimension via average pooling
  weight = convolution(X=X, kernel_shape=(3, 3), n_filters=filter_in * filter_out, stride=(1, 1), pad=(1, 1))
  weight = ReLU(weight)
  pad = {8 : (2, 2), 16 : (1, 1), 32 : (2, 2)}[H]
  kernel_shape = {8 : (4, 4), 16 : (6, 6), 32 : (12, 12)}[H]
  stride = kernel_shape
  weight = pooling(X=weight, mode='average', kernel_shape=kernel_shape, stride=stride, pad=pad)
  return weight

_convolution_embedding_count = 0
def convolution_weight_from_parameters(embedding, d, filter_in, filter_out, width, height):
  # implementing the method proposed in "HyperNetworks"
  global _convolution_embedding_count
  if isinstance(embedding, int):
    weight_id = 'convolution_embedding%d_weight' % _convolution_embedding_count
    weight = variable(weight_id, shape=(1, embedding))
  else: weight = embedding
  _convolution_embedding_count += 1
  weight = fully_connected(X=weight, n_hidden_units=filter_in * d)
  weight = reshape(weight, (filter_in, d))
  weight = fully_connected(X=weight, n_hidden_units=filter_out * width * height)
  weight = reshape(weight, (filter_in, filter_out, width, height))
  weight = swap_axes(weight, 0, 1)
  return weight

if __name__ is '__main__':
  EMBEDDING = 64
  FILTER_IN, FILTER_OUT = 3, 16
  WIDTH, HEIGHT = 3, 3
  weight = convolution_weight_from_parameters(EMBEDDING, EMBEDDING, FILTER_IN, FILTER_OUT, WIDTH, HEIGHT)
  arg_shapes, output_shapes, _ = weight.infer_shape(convolution_embedding0_weight=(1, EMBEDDING))
  print output_shapes[0]

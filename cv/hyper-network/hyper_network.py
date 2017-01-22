import mxnet as mx
from mx_layers import *

def generated_convolution_embedding(X, kernel, batch_size):
  # method 0
  X = pooling(X, 'average', kernel, (1, 1))
  X = flatten(X)
  X = mx.symbol.sum(X, axis=0) / float(batch_size)
  return X

def convolution_weight_from_feature_maps(X, filter_in, filter_out, data_shape):
  # dynamic filter network
  # limited purpose
  _, X_shape, _ = X.infer_shape(data=data_shape)
  N, C, H, W = X_shape[0]
  weight = convolution(X=X, kernel_shape=(3, 3), n_filters=filter_in * filter_out, stride=(1, 1), pad=(2, 2))
  weight = ReLU(weight)
  pad = {8 : (2, 2), 16 : (1, 1), 32 : (2, 2)}[H]
  stride = {8 : (4, 4), 16 : (6, 6), 32 : (12, 12)}[H]
  kernel_shape = stride
  weight = pooling(X=weight, mode='maximum', kernel_shape=kernel_shape, stride=stride, pad=pad)
  return weight

_convolution_embedding_count = 0
def convolution_weight_from_parameters(embedding, d, filter_in, filter_out, width, height):
  # implementing the method proposed in "HyperNetworks"
  global _convolution_embedding_count
  if isinstance(embedding, int): # use postfix to inform intializer
    weight = variable('convolution_embedding%d_weight' % _convolution_embedding_count, shape=(embedding,))
  else: weight = embedding
  _convolution_embedding_count += 1
  weight = fully_connected(X=weight, n_hidden_units=filter_in * d)
  weight = reshape(weight, (filter_in, d))
  weight = fully_connected(X=weight, n_hidden_units=filter_out * width * height)
  weight = reshape(weight, (filter_in, filter_out, width, height))
  weight = swap_axes(weight, 0, 1)
  return weight

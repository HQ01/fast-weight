import mxnet as mx
from mx_layers import *
import numpy as np0

from facility import mark

N, C, W, H = 1, 3, 32, 32
X_SHAPE = (N, C, W, H)
network = mx.symbol.Variable('X')
network = convolution(X=network, kernel_shape=(3, 3), n_filters=9, stride=1, pad=1, n_groups=N)
network = fully_connected(X=network, n_hidden_units=10)
network = softmax_loss(network)
arguments = network.list_arguments()
argument_shapes, output_shapes, _ = network.infer_shape(X=X_SHAPE)
print dict(zip(arguments, argument_shapes))

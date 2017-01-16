import mxnet as mx
from mx_layers import *
import numpy as np0

from facility import mark

X_SHAPE = (27, 3, 32, 32)
X = variable('X', shape=X_SHAPE)
Y = convolution(X=X, kernel_shape=(3, 3), n_filters=9, stride=1, pad=1)
arg_shapes, output_shapes, _ = Y.infer_shape(X=X_SHAPE)
print arg_shapes

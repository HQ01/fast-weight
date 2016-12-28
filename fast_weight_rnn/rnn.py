# TODO initialization (paper)
# TODO character embedding

import mxnet
import minpy.nn.layers as layers
from minpy.nn.model import ModelBase
import minpy.numpy as np
import minpy.numpy.random as random
import numpy as np0

from facility import *

class FastWeightRNN(ModelBase):
  def __init__(self, input_size, n_hidden, n_classes, inner_length):
    super(FastWeightRNN, self).__init__()
    self._inner_length = inner_length
    self._input_size = input_size
    self._n_hidden = n_hidden
    self._n_classes = n_classes

    self._decay_rate = 0.95
    self._learning_rate = 0.5
    self._nonlinear = np.tanh

    self \
      .add_param(
        name        = 'WX',
        shape       = (input_size, n_hidden),
        init_rule   = 'custom',
        init_config = {
          'function' : lambda shape : np.random.uniform(-n_hidden ** 0.5, n_hidden ** 0.5, shape)
        }
      ) \
      .add_param(
        name        = 'Wh',
        shape       = (n_hidden, n_hidden),
        init_rule   = 'custom',
        init_config = {'function' : lambda shape : np0.identity(n_hidden)}
      ) \
      .add_param(
        name        = 'bias_h',
        shape       = (n_hidden,),
        init_rule   = 'constant',
        init_config = {'value' : 0}
      ) \
      .add_param(
        name        = 'gamma',
        shape       = (n_hidden,),
        init_rule   = 'constant',
        init_config = {'value' : 1}
      ) \
      .add_param(
        name        = 'beta',
        shape       = (n_hidden,),
        init_rule   = 'constant',
        init_config = {'value' : 0}
      ) \
      .add_param(
        name        = 'WY0',
        shape       = (n_hidden, 100),
        init_rule   = 'xavier',
        init_config = {}
      )\
      .add_param(
        name        = 'bias_Y0',
        shape       = (100,),
        init_rule   = 'constant',
        init_config = {'value' : 0}
      ) \
      .add_param(
        name        = 'WY',
        shape       = (100, n_classes),
        init_rule   = 'xavier',
        init_config = {}
      )\
      .add_param(
        name        = 'bias_Y',
        shape       = (n_classes,),
        init_rule   = 'constant',
        init_config = {'value' : 0}
      )

  def _update_h(self, X, previous_h, WX, Wh, bias):
    next_h = self._nonlinear(np.dot(X, WX) + np.dot(previous_h, Wh) + bias)
    return next_h

  def _inner_loop(self, X, h, h0, WX, Wh, previous_h):
    # TODO efficiency
    N, H = h.shape
    gamma, beta = self.params['gamma'], self.params['beta']
    hs = h0
    for s in xrange(self._inner_length):
      projected_hs = self._learning_rate * sum(
        self._decay_rate ** (len(previous_h) - t - 1) * diagonal(np.dot(h, hs.T)) * h
          for t, h in enumerate(previous_h)
      )
      hs = np.dot(X, WX) + np.dot(h, Wh) + projected_hs
      hs = layer_normalization(hs, gamma, beta)
      hs = self._nonlinear(hs)
    return hs

  def forward(self, X, mode):
    N, sequence_length, D = X.shape
    h = np.zeros((N, self._n_hidden))

    WX      = self.params['WX']
    Wh      = self.params['Wh']
    bias_h  = self.params['bias_h']
    WY      = self.params['WY']
    bias_Y  = self.params['bias_Y']
    WY0     = self.params['WY0']
    bias_Y0 = self.params['bias_Y0']

    previous_h = [h]
    for t in xrange(sequence_length):
      h = self._update_h(X[:, t, :], h, WX, Wh, bias_h)
      h = self._inner_loop(X[:, t, :], previous_h[-1], h, WX, Wh, previous_h)
      previous_h.append(h)

    Y0 = layers.relu(layers.affine(h, WY, bias_Y))
    Y = layers.affine(Y0, WY, bias_Y)
    return Y

  def loss(self, prediction, Y):
    return layers.softmax_loss(prediction, Y)

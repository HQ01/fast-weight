# TODO initialization (paper)
# TODO character embedding

import mxnet
import minpy.nn.layers as layers
from minpy.nn.model import ModelBase
import minpy.numpy as np
import minpy.numpy.random as random

from custom_layers import *
from facility import *

class FastWeightRNN(ModelBase):
  def __init__(self, input_size, n_hidden, n_classes, inner_length):
    super(FastWeightRNN, self).__init__()
    self._inner_length = inner_length
    self._input_size = input_size
    self._n_hidden = n_hidden
    self._n_classes = n_classes

    self._learning_rate = 0.5
    self._decay_rate = 0.5
#   self._learning_rate = 0.5
#   self._decay_rate = 0.9
    self._nonlinear = np.tanh

    self \
      .add_param(
        name        = 'WX',
        shape       = (input_size, n_hidden),
        init_rule   = 'xavier',
        init_config = {}
#        init_rule   = 'custom',
#        init_config = {
#          'function' : lambda shape : np.random.uniform(-n_hidden ** 0.5, n_hidden ** 0.5, shape)
#        }
      ) \
      .add_param(
        name        = 'Wh',
        shape       = (n_hidden, n_hidden),
        init_rule   = 'custom',
        init_config = {'function' : lambda shape : identity(n_hidden) * 0.01}
#       init_config = {'function' : lambda shape : identity(n_hidden) * 0.05}
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
    boundary_condition = np.dot(X, WX) + np.dot(h, Wh)
    hs = h0
    for s in xrange(self._inner_length):
      '''
      projected_hs = self._learning_rate * sum(
        self._decay_rate ** (len(previous_h) - t - 1) * np.dot(diagonal(np.dot(h, hs.T)), h)
          for t, h in enumerate(previous_h)
      )
      '''
      projected_hs = self._learning_rate * sum(
        self._decay_rate ** (len(previous_h) - t - 1) * batch_scalar_product(h, hs) * h
          for t, h in enumerate(previous_h)
      )
      print 'boundary_condition', np.min(boundary_condition), np.max(boundary_condition)
      print 'projected_hs', np.min(projected_hs), np.max(projected_hs)
      '''
      raise Exception()
      '''
      hs = boundary_condition + projected_hs
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

    self.previous_h = [h]
    for t in xrange(sequence_length):
      X_t = X[:, t, :]
      h = self._update_h(X_t, h, WX, Wh, bias_h)
      h = self._inner_loop(X_t, self.previous_h[-1], h, WX, Wh, self.previous_h)
      self.previous_h.append(h)

    Y0 = layers.relu(layers.affine(h, WY0, bias_Y0))
    Y = layers.affine(Y0, WY, bias_Y)
    return Y

  def loss(self, prediction, Y):
    return layers.softmax_loss(prediction, Y)

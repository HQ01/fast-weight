import mxnet
import minpy.nn.layers as layers
from minpy.nn.model import ModelBase
import minpy.numpy as np
import minpy.numpy.random as random

from minpy.context import set_context, gpu
set_context(gpu(0))

from solver_primitives import *

class RNNNet(ModelBase):
  def __init__(self, input_size, n_hidden, n_classes):
    super(RNNNet, self).__init__()
    self._input_size = input_size
    self._n_hidden = n_hidden
    self._n_classes = n_classes

    self._decay_rate = 0.95
    self._learning_rate = 0.5
    self._inner_length = 5
    self._nonlinear = np.tanh

    self \
      .add_param(name='WX', shape=(input_size, n_hidden)) \
      .add_param(
        name        = 'Wh',
        shape       = (n_hidden, n_hidden),
        init_rule   = 'constant',
        init_config = {'value' : np.identity(n_hidden)}
      ) \
      .add_param(
        name        ='b',
        shape       = (n_hidden,),
        init_rule   = 'constant',
        init_config = {'value' : np.zeros(n_hidden)}
      ) \
      .add_param(name='WY', shape=(n_hidden, n_classes))\
      .add_param(name='bias_Y', shape=(n_classes,)) \
      .add_param(
        name        = 'A',
        shape       = (n_hidden, n_hidden),
        init_rule   = 'constant',
        init_config = {'value' : np.random.uniform(0.0, 0.01, (n_hidden, n_hidden))}
      )

  def _step(self, X, previous_h, WX, Wh, bias):
    next_h = self._nonlinear(np.dot(X, WX) + np.dot(previous_h, Wh) + bias)
    return next_h

  def _inner_loop(self, X, h, WX, Wh, A):
    for t in xrange(self._inner_length):
      h = self._nonlinear(np.dot(X, WX) + np.dot(h, Wh) + np.dot(h, A))
    return h
  def forward(self, X, mode):
    N, sequence_length, D = X.shape
    h = np.zeros((N, self._n_hidden))
    A = 0 # TODO whether A is a parameter

    WX     = self.params['WX']
    Wh     = self.params['Wh']
    bias   = self.params['b']
    WY     = self.params['WY']
    bias_Y = self.params['bias_Y']

    for t in xrange(sequence_length):
      h = self._step(X[:, t, :], h, WX, Wh, bias)
      A = self._decay_rate * A + self._learning_rate * np.outer(h.T, h)
      h = self._inner_loop(X[:, t, :], h, WX, Wh, A)

    return layers.affine(h, WY, bias_Y)

  def loss(self, prediction, Y):
    return layers.softmax_loss(prediction, Y)

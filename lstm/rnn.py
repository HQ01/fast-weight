import mxnet
import minpy.nn.layers as layers
from minpy.nn.model import ModelBase
import minpy.numpy as np
import minpy.numpy.random as random
import numpy as np0

from facility import *

class LSTMRNN(ModelBase):
  def __init__(self, input_size, n_hidden, n_classes):
    super(LSTMRNN, self).__init__()
    self._input_size = input_size
    self._n_hidden = n_hidden
    self._n_classes = n_classes
    # TODO initialization
    self \
      .add_param(
        name        = 'WX',
        shape       = (input_size, 4 * n_hidden),
        init_rule   = 'custom',
        init_config = {
          'function' : lambda shape : np.random.uniform(-n_hidden ** 0.5, n_hidden ** 0.5, shape)
        }
      ) \
      .add_param(
        name        = 'Wh',
        shape       = (n_hidden, 4 * n_hidden),
        init_rule   = 'xavier',
        init_config = {}
      ) \
      .add_param(
        name        = 'bias',
        shape       = (4 * n_hidden,),
        init_rule   = 'constant',
        init_config = {'value' : 0}
      ) \
      .add_param(
        name        = 'WY',
        shape       = (n_hidden, n_classes),
        init_rule   = 'xavier',
        init_config = {}
      ) \
      .add_param(
        name        = 'bias_Y',
        shape       = (n_classes,),
        init_rule   = 'constant',
        init_config = {'value' : 0}
      )

  def forward(self, X, mode):
    N, sequence_length, D = X.shape
    h = np.zeros((N, self._n_hidden))
    c = np.zeros((N, self._n_hidden))

    WX     = self.params['WX']
    Wh     = self.params['Wh']
    bias   = self.params['bias']
    WY     = self.params['WY']
    bias_Y = self.params['bias_Y']

    for t in range(sequence_length):
      X_t = X[:, t, :]
      h, c = layers.lstm_step(X_t, h, c, WX, Wh, bias)

    Y = layers.affine(h, WY, bias_Y)
    return Y

  def loss(self, prediction, Y):
    return layers.softmax_loss(prediction, Y)

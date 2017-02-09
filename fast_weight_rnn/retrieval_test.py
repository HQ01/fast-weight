import mxnet.symbol as symbol
from minpy.core import Function
from minpy.core import grad_and_loss as gradient_loss
import minpy.nn.init as initializers
import minpy.nn.layers as layers
from minpy.nn.model import ModelBase
import minpy.numpy as np

from minpy.context import set_context, cpu
set_context(cpu())

def batch_dot(left, right):
  # wraps mxnet.symbol.batch_dot
  left_symbol = symbol.Variable('left')
  right_symbol = symbol.Variable('right')
  result_symbol = symbol.batch_dot(left_symbol, right_symbol)
  shapes = {'left' : left.shape, 'right' : right.shape}
  kwargs = {'left' : left, 'right' : right}
  return Function(result_symbol, shapes)(**kwargs)

def batch_scalar_product(left, right):
  # helper function
  N, D = left.shape
  left = left.reshape((N, 1, D))
  right = right.reshape((N, D ,1))
  result = batch_dot(left, right)
  result = result.reshape((N, 1))
  return result

class RNN(ModelBase):
  def __init__(self):
    super(RNN, self).__init__()
    self._input_size = 128
    self._n_hidden = 20
    self._n_classes = 10
    self._nonlinear = np.tanh

    self \
      .add_param(
        name        = 'WX',
        shape       = (self._input_size, self._n_hidden),
        init_rule   = 'custom',
        init_config = {
          'function' : lambda shape : np.random.uniform(-self._n_hidden ** 0.5, self._n_hidden ** 0.5, shape)
        }
      ) \
      .add_param(
        name        = 'Wh',
        shape       = (self._n_hidden, self._n_hidden),
        init_rule   = 'custom',
        init_config = {'function' : lambda shape : np.eye(self._n_hidden) * 0.05}
      ) \
      .add_param(
        name        = 'bias_h',
        shape       = (self._n_hidden,),
        init_rule   = 'constant',
        init_config = {'value' : 0}
      ) \
      .add_param(
        name        = 'WY0',
        shape       = (self._n_hidden, 100),
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
        shape       = (100, self._n_classes),
        init_rule   = 'xavier',
        init_config = {}
      )\
      .add_param(
        name        = 'bias_Y',
        shape       = (self._n_classes,),
        init_rule   = 'constant',
        init_config = {'value' : 0}
      )

  def _update_h(self, X, previous_h, WX, Wh, bias):
    next_h = self._nonlinear(np.dot(X, WX) + np.dot(previous_h, Wh) + bias)
    return next_h

  def forward(self, X, mode):
    N, sequence_length, D = X.shape
    WX      = self.params['WX']
    Wh      = self.params['Wh']
    bias_h  = self.params['bias_h']
    WY      = self.params['WY']
    bias_Y  = self.params['bias_Y']
    WY0     = self.params['WY0']
    bias_Y0 = self.params['bias_Y0']

    h = np.zeros((N, self._n_hidden))
    self.previous_h = [h]
    for t in xrange(sequence_length):
      X_t = X[:, t, :]
      h0 = self._update_h(X_t, h, WX, Wh, bias_h)
      projected_h = sum(batch_scalar_product(h, h0) * h for t, h in enumerate(self.previous_h))
      h = np.dot(X_t, WX) + np.dot(h, Wh) + projected_h
      h = self._nonlinear(h)
      self.previous_h.append(h)

    Y0 = layers.relu(layers.affine(h, WY0, bias_Y0))
    Y = layers.affine(Y0, WY, bias_Y)
    return Y

  def loss(self, prediction, Y):
    return layers.softmax_loss(prediction, Y)

model = RNN()
# initialize model
for key, value in model.param_configs.items():
  model.params[key] = getattr(initializers, value['init_rule'])(
    value['shape'],
    value.get('init_config', {})
  )

N = 64
X, Y = np.ones((N, 11, 128)), np.ones((N,))

def loss_function(X, Y, *args):
  predictions = model.forward(X, 'train')
  return model.loss(predictions, Y)
gl = gradient_loss(loss_function, range(2, len(model.params) + 2))
g, loss = gl(X, Y, *list(model.params.values()))

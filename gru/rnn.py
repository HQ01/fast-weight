import numpy as np

# from facility import sigmoid
def sigmoid(X):
  return 1 / (1 + np.exp(-X))

def gru_step(X, h, WX, Uh, W, U):
  N, H = h.shape
  activation = sigmoid(np.dot(X, W) + np.dot(h, U))
  r = activation[:, 0 : H]
  _h = np.tanh(np.dot(X, WX) + np.dot(h * r, Uh))
  z = activation[:, H : 2 * H]
  h = (1 - z) * _h + z * h
  return h

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
        shape       = (input_size, n_hidden),
        init_rule   = 'custom',
        init_config = {
          'function' : lambda shape : np.random.uniform(-n_hidden ** 0.5, n_hidden ** 0.5, shape)
        }
      ) \

